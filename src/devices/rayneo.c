#include "devices.h"
#include "devices/rayneo.h"
#include "connection_pool.h"
#include "driver.h"
#include "imu.h"
#include "logging.h"
#include "outputs.h"
#include "runtime_context.h"

#include <libusb.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// This driver speaks the RayNeo USB protocol directly (no proprietary SDK). The protocol and
// IMU fusion were reverse-engineered from the RayNeo-Air-3S-Pro-OpenVR project and a companion
// Android reference implementation.
//
// Transport: 64-byte HID-style frames over a USB interrupt (or bulk) interface.
//   - Outgoing command frames begin with 0x66: [0x66][cmd][arg][0...].
//   - Incoming frames begin with 0x99: [0x99][type][len]...; type 0x65 = IMU sample, 0xC8 = device info.
// Fusion: a complementary filter (gyro integration corrected toward the accelerometer's gravity
// vector, with stationary gyro-bias estimation) produces an orientation quaternion.

#define RAYNEO_DRIVER_ID "rayneo"

#define RAYNEO_ID_VENDOR 0x1bbb
#define RAYNEO_ID_PRODUCT 0xaf50

// Outgoing command frame
#define FRAME_CMD_MAGIC 0x66
#define CMD_ACQUIRE_DEVICE_INFO 0x00
#define CMD_OPEN_IMU 0x01
#define CMD_CLOSE_IMU 0x02
#define CMD_SWITCH_TO_3D 0x06
#define CMD_SWITCH_TO_2D 0x07

// Incoming frame
#define FRAME_IN_MAGIC 0x99
#define FRAME_TYPE_SENSOR 0x65
#define FRAME_TYPE_RESPONSE 0xC8

// IMU sample byte offsets (relative to start of the 64-byte frame)
#define SENSOR_ACCEL_OFFSET 4   // 3x float32 LE, m/s^2
#define SENSOR_GYRO_OFFSET 16   // 3x float32 LE, degrees/second
#define SENSOR_TICK_OFFSET 40   // uint32 LE, units of 100us
#define FRAME_LEN 64

// Device info byte offsets (relative to start of the 64-byte frame)
#define DEVINFO_BOARD_ID_OFFSET 21
#define DEVINFO_SIDE_BY_SIDE_OFFSET 43

// Known board ids
#define BOARD_AIR_4_PRO 0x3A

#define EXPECTED_CYCLES_PER_S 500
#define FORCED_CYCLES_PER_S 250 // throttle pose ingestion to reduce downstream computation
#define CYCLE_TIME_CHECK_ERROR_FACTOR 0.95 // cycle times won't be exact, check within a 5% margin
#define FORCED_CYCLE_TIME_MS (1000.0 / FORCED_CYCLES_PER_S * CYCLE_TIME_CHECK_ERROR_FACTOR)
#define BUFFER_SIZE_TARGET_MS 10 // smooth IMU data over this period of time

// The complementary filter produces orientation in a Y-up world frame (gravity aligned to +Y).
// The driver pipeline expects a Z-up "north-west-up" frame (yaw about Z). A +90-degree rotation
// about X maps the filter's +Y (up) onto the driver's +Z (up).
//
// NOTE: this maps the gravity (up) axis correctly, which makes pitch/roll absolute and leaves yaw
// as the free axis that the driver recenters. If, on-device, pitch and roll appear swapped or a
// rotation direction feels inverted, this is the constant to tune.
static const imu_quat_type filter_to_nwu_quat = {
    .w = 0.70710678f,
    .x = 0.70710678f,
    .y = 0.0f,
    .z = 0.0f
};

const device_properties_type rayneo_properties = {
    .brand                              = "RayNeo",
    .model                              = "Air",
    .hid_vendor_id                      = RAYNEO_ID_VENDOR,
    .hid_product_id                     = RAYNEO_ID_PRODUCT,
    .calibration_setup                  = CALIBRATION_SETUP_AUTOMATIC,
    .resolution_w                       = RESOLUTION_1080P_W,
    .resolution_h                       = RESOLUTION_1080P_H,
    .fov                                = 43.0,
    .lens_distance_ratio                = 0.05,
    .calibration_wait_s                 = 5,
    .imu_cycles_per_s                   = FORCED_CYCLES_PER_S,
    .imu_buffer_size                    = ceil(BUFFER_SIZE_TARGET_MS / FORCED_CYCLE_TIME_MS),
    .look_ahead_constant                = 15.0,
    .look_ahead_frametime_multiplier    = 0.45,
    .look_ahead_scanline_adjust         = 12.0,
    .look_ahead_ms_cap                  = 40.0,
    .sbs_mode_supported                 = true,
    .firmware_update_recommended        = false,
    .provides_orientation               = true,
    .provides_position                  = false
};

// ---------------------------------------------------------------------------
// USB transport state (guarded by usb_mutex)
// ---------------------------------------------------------------------------
static pthread_mutex_t usb_mutex = PTHREAD_MUTEX_INITIALIZER;
static libusb_context* usb_ctx = NULL;
static libusb_device_handle* usb_handle = NULL;
static int usb_interface_num = -1;
static uint8_t ep_in = 0;
static uint8_t ep_out = 0;
static bool ep_in_is_interrupt = true;
static bool ep_out_is_interrupt = true;
static int ep_in_max_packet = FRAME_LEN;

// hardware connection - USB handle is open
static bool hard_connected = false;
// software connection - we're actively streaming IMU data
static bool soft_connected = false;

static volatile bool is_sbs_mode = false;
static uint8_t device_board_id = 0;
static float device_imu_rotation_x_deg = 0.0f;

// ---------------------------------------------------------------------------
// Orientation filter (complementary: gyro integration + accelerometer correction)
// ---------------------------------------------------------------------------
typedef struct {
    imu_quat_type q;            // current orientation, filter (Y-up) frame
    imu_vec3_type gyro_bias;    // estimated gyro bias, rad/s
    imu_quat_type imu_rotation; // fixed correction for the IMU's mounting tilt
    int64_t last_tick_100us;    // device tick of previous sample (-1 if none)
    int64_t last_realtime_ns;   // host time of previous sample (fallback dt source)
} orientation_filter_type;

static orientation_filter_type filter;

static const imu_vec3_type WORLD_UP = { .x = 0.0f, .y = 1.0f, .z = 0.0f };
static const float ACCEL_GAIN = 1.5f;
static const float GRAVITY_MPS2 = 9.81f;
static const float STATIONARY_ACCEL_TOL = 1.25f;
static const float STATIONARY_GYRO_RAD_PER_SEC = 0.18f;
static const float GYRO_BIAS_UPDATE_HZ = 0.5f;

static int64_t monotonic_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static float vec3_length(imu_vec3_type v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

static imu_vec3_type vec3_cross(imu_vec3_type a, imu_vec3_type b) {
    imu_vec3_type out = {
        .x = a.y * b.z - a.z * b.y,
        .y = a.z * b.x - a.x * b.z,
        .z = a.x * b.y - a.y * b.x
    };
    return out;
}

// integrate body angular rate (rad/s) into orientation q over dt seconds:
//   q_dot = 0.5 * q * (0, omega); q' = normalize(q + q_dot * dt)
static imu_quat_type quat_integrate_body_rate(imu_quat_type q, imu_vec3_type omega, float dt) {
    // raw (non-normalizing) multiply q * (0, omega)
    imu_quat_type qd = {
        .w = -q.x * omega.x - q.y * omega.y - q.z * omega.z,
        .x =  q.w * omega.x + q.y * omega.z - q.z * omega.y,
        .y =  q.w * omega.y - q.x * omega.z + q.z * omega.x,
        .z =  q.w * omega.z + q.x * omega.y - q.y * omega.x
    };
    imu_quat_type out = {
        .w = q.w + 0.5f * qd.w * dt,
        .x = q.x + 0.5f * qd.x * dt,
        .y = q.y + 0.5f * qd.y * dt,
        .z = q.z + 0.5f * qd.z * dt
    };
    return normalize_quaternion(out);
}

static imu_quat_type imu_rotation_quat_from_deg(float imu_rotation_x_deg) {
    float half = degree_to_radian(imu_rotation_x_deg) * 0.5f;
    return (imu_quat_type){ .w = cosf(half), .x = sinf(half), .y = 0.0f, .z = 0.0f };
}

static void filter_reset(orientation_filter_type* f, float imu_rotation_x_deg) {
    f->q = (imu_quat_type){ .w = 1.0f, .x = 0.0f, .y = 0.0f, .z = 0.0f };
    f->gyro_bias = (imu_vec3_type){ 0.0f, 0.0f, 0.0f };
    f->imu_rotation = imu_rotation_quat_from_deg(imu_rotation_x_deg);
    f->last_tick_100us = -1;
    f->last_realtime_ns = 0;
}

// feed one raw sample (accel m/s^2, gyro degrees/s, device tick in 100us units) into the filter
static imu_quat_type filter_update(orientation_filter_type* f, imu_vec3_type accel, imu_vec3_type gyro_dps,
                                   uint32_t tick_100us) {
    imu_vec3_type accel_b = vector_rotate(accel, f->imu_rotation);
    imu_vec3_type gyro = {
        .x = degree_to_radian(gyro_dps.x),
        .y = degree_to_radian(gyro_dps.y),
        .z = degree_to_radian(gyro_dps.z)
    };
    gyro = vector_rotate(gyro, f->imu_rotation);

    int64_t now_ns = monotonic_ns();
    float dt = 0.01f;
    if (f->last_tick_100us >= 0 && (int64_t) tick_100us > f->last_tick_100us) {
        dt = (float)((int64_t) tick_100us - f->last_tick_100us) * 1e-4f;
    } else if (f->last_realtime_ns > 0 && now_ns > f->last_realtime_ns) {
        dt = (float)(now_ns - f->last_realtime_ns) * 1e-9f;
    }
    if (dt < 0.001f || dt > 0.1f) dt = 0.01f;
    f->last_tick_100us = (int64_t) tick_100us;
    f->last_realtime_ns = now_ns;

    float accel_norm = vec3_length(accel_b);
    bool stationary = fabsf(accel_norm - GRAVITY_MPS2) < STATIONARY_ACCEL_TOL &&
                      vec3_length(gyro) < STATIONARY_GYRO_RAD_PER_SEC;
    if (stationary) {
        float alpha = dt * GYRO_BIAS_UPDATE_HZ;
        if (alpha > 1.0f) alpha = 1.0f;
        f->gyro_bias.x = f->gyro_bias.x * (1.0f - alpha) + gyro.x * alpha;
        f->gyro_bias.y = f->gyro_bias.y * (1.0f - alpha) + gyro.y * alpha;
        f->gyro_bias.z = f->gyro_bias.z * (1.0f - alpha) + gyro.z * alpha;
    }
    gyro.x -= f->gyro_bias.x;
    gyro.y -= f->gyro_bias.y;
    gyro.z -= f->gyro_bias.z;

    if (accel_norm > 1e-3f) {
        imu_vec3_type measured_up = { accel_b.x / accel_norm, accel_b.y / accel_norm, accel_b.z / accel_norm };
        imu_vec3_type predicted_up = vector_rotate(WORLD_UP, conjugate(f->q));
        float pu_norm = vec3_length(predicted_up);
        if (pu_norm > 1e-6f) {
            predicted_up.x /= pu_norm; predicted_up.y /= pu_norm; predicted_up.z /= pu_norm;
        }
        imu_vec3_type correction = vec3_cross(predicted_up, measured_up);
        gyro.x += correction.x * ACCEL_GAIN;
        gyro.y += correction.y * ACCEL_GAIN;
        gyro.z += correction.z * ACCEL_GAIN;
    }

    f->q = quat_integrate_body_rate(f->q, gyro, dt);
    return f->q;
}

// ---------------------------------------------------------------------------
// USB helpers (caller must hold usb_mutex)
// ---------------------------------------------------------------------------
// Choose the interface used for IMU/control traffic. RayNeo glasses expose several USB
// interfaces (video, audio, HID/control); the sensor + command interface uses interrupt
// endpoints, so prefer an interface whose IN and OUT endpoints are both interrupt, and fall
// back to the first interface that has any IN+OUT endpoint pair. Mirrors the Android reference.
static bool rayneo_select_interface(libusb_device* dev) {
    struct libusb_config_descriptor* cfg = NULL;
    if (libusb_get_active_config_descriptor(dev, &cfg) != 0 || !cfg) return false;

    bool found = false;       // selected a preferred (interrupt IN+OUT) interface
    bool have_fallback = false;
    int fb_iface = -1, fb_in_max = FRAME_LEN;
    uint8_t fb_in = 0, fb_out = 0;
    bool fb_in_int = false, fb_out_int = false;

    for (uint8_t i = 0; i < cfg->bNumInterfaces && !found; i++) {
        const struct libusb_interface* iface = &cfg->interface[i];
        for (int a = 0; a < iface->num_altsetting && !found; a++) {
            const struct libusb_interface_descriptor* id = &iface->altsetting[a];
            uint8_t in_ep = 0, out_ep = 0;
            bool in_interrupt = false, out_interrupt = false;
            int in_max = FRAME_LEN;
            for (uint8_t e = 0; e < id->bNumEndpoints; e++) {
                const struct libusb_endpoint_descriptor* epd = &id->endpoint[e];
                uint8_t type = epd->bmAttributes & 0x03;
                if (type == LIBUSB_TRANSFER_TYPE_CONTROL) continue;
                bool is_interrupt = (type == LIBUSB_TRANSFER_TYPE_INTERRUPT);
                if ((epd->bEndpointAddress & LIBUSB_ENDPOINT_DIR_MASK) == LIBUSB_ENDPOINT_IN) {
                    if (in_ep == 0) {
                        in_ep = epd->bEndpointAddress;
                        in_interrupt = is_interrupt;
                        in_max = epd->wMaxPacketSize ? epd->wMaxPacketSize : FRAME_LEN;
                    }
                } else if (out_ep == 0) {
                    out_ep = epd->bEndpointAddress;
                    out_interrupt = is_interrupt;
                }
            }
            if (in_ep != 0 && out_ep != 0) {
                if (in_interrupt && out_interrupt) {
                    usb_interface_num = id->bInterfaceNumber;
                    ep_in = in_ep;
                    ep_out = out_ep;
                    ep_in_is_interrupt = true;
                    ep_out_is_interrupt = true;
                    ep_in_max_packet = in_max;
                    found = true;
                } else if (!have_fallback) {
                    fb_iface = id->bInterfaceNumber;
                    fb_in = in_ep;
                    fb_out = out_ep;
                    fb_in_int = in_interrupt;
                    fb_out_int = out_interrupt;
                    fb_in_max = in_max;
                    have_fallback = true;
                }
            }
        }
    }

    if (!found && have_fallback) {
        usb_interface_num = fb_iface;
        ep_in = fb_in;
        ep_out = fb_out;
        ep_in_is_interrupt = fb_in_int;
        ep_out_is_interrupt = fb_out_int;
        ep_in_max_packet = fb_in_max;
        found = true;
    }

    if (found && config()->debug_device) {
        log_debug("RayNeo driver, selected interface %d, ep_in 0x%02x (%s), ep_out 0x%02x (%s)\n",
                  usb_interface_num, ep_in, ep_in_is_interrupt ? "interrupt" : "bulk",
                  ep_out, ep_out_is_interrupt ? "interrupt" : "bulk");
    }

    libusb_free_config_descriptor(cfg);
    return found;
}

static bool rayneo_open_usb(uint8_t usb_bus, uint8_t usb_address) {
    if (usb_handle) return true;
    if (!usb_ctx && libusb_init(&usb_ctx) != 0) {
        log_error("RayNeo driver, failed to initialize libusb\n");
        return false;
    }

    libusb_device** list = NULL;
    ssize_t count = libusb_get_device_list(usb_ctx, &list);
    if (count < 0) return false;

    libusb_device* target = NULL;
    for (ssize_t i = 0; i < count; i++) {
        struct libusb_device_descriptor desc;
        if (libusb_get_device_descriptor(list[i], &desc) != 0) continue;
        if (desc.idVendor != RAYNEO_ID_VENDOR || desc.idProduct != RAYNEO_ID_PRODUCT) continue;
        // prefer the exact device the hotplug layer told us about
        if (usb_bus != 0 &&
            (libusb_get_bus_number(list[i]) != usb_bus || libusb_get_device_address(list[i]) != usb_address)) {
            if (target == NULL) target = list[i]; // fall back to first match
            continue;
        }
        target = list[i];
        break;
    }

    bool ok = false;
    if (target == NULL) {
        log_error("RayNeo driver, device 0x%04x:0x%04x not found on the USB bus\n",
                  RAYNEO_ID_VENDOR, RAYNEO_ID_PRODUCT);
    } else {
        int rc = libusb_open(target, &usb_handle);
        if (rc != 0 || !usb_handle) {
            log_error("RayNeo driver, failed to open USB device: %s\n", libusb_error_name(rc));
            usb_handle = NULL;
        } else if (rayneo_select_interface(libusb_get_device(usb_handle))) {
            libusb_set_auto_detach_kernel_driver(usb_handle, 1);
            if (libusb_kernel_driver_active(usb_handle, usb_interface_num) == 1) {
                libusb_detach_kernel_driver(usb_handle, usb_interface_num);
            }
            int crc = libusb_claim_interface(usb_handle, usb_interface_num);
            if (crc == 0) {
                ok = true;
            } else {
                log_error("RayNeo driver, failed to claim USB interface %d: %s\n",
                          usb_interface_num, libusb_error_name(crc));
            }
        } else {
            log_error("RayNeo driver, could not find usable IN/OUT endpoints\n");
        }
        if (!ok && usb_handle) {
            libusb_close(usb_handle);
            usb_handle = NULL;
            usb_interface_num = -1;
            ep_in = ep_out = 0;
        }
    }

    libusb_free_device_list(list, 1);
    return ok;
}

static void rayneo_close_usb() {
    if (usb_handle) {
        if (usb_interface_num >= 0) libusb_release_interface(usb_handle, usb_interface_num);
        libusb_close(usb_handle);
        usb_handle = NULL;
    }
    usb_interface_num = -1;
    ep_in = ep_out = 0;
}

static bool rayneo_send(uint8_t cmd, uint8_t arg) {
    if (!usb_handle || !ep_out) return false;
    uint8_t frame[FRAME_LEN] = {0};
    frame[0] = FRAME_CMD_MAGIC;
    frame[1] = cmd;
    frame[2] = arg;

    int transferred = 0;
    int rc = ep_out_is_interrupt
        ? libusb_interrupt_transfer(usb_handle, ep_out, frame, FRAME_LEN, &transferred, 500)
        : libusb_bulk_transfer(usb_handle, ep_out, frame, FRAME_LEN, &transferred, 500);
    if (rc == 0 && transferred == FRAME_LEN) return true;

    // fallback: HID SET_REPORT (Output report, report id 0)
    rc = libusb_control_transfer(usb_handle, 0x21, 0x09, (0x02 << 8) | 0x00,
                                 usb_interface_num >= 0 ? usb_interface_num : 0, frame, FRAME_LEN, 500);
    return rc == FRAME_LEN;
}

// returns: >=0 bytes read (0 on timeout), or a negative libusb error
static int rayneo_read_frame(uint8_t* buf, int buf_size, int timeout_ms) {
    if (!usb_handle || !ep_in) return LIBUSB_ERROR_NO_DEVICE;
    int length = ep_in_max_packet > 0 ? ep_in_max_packet : FRAME_LEN;
    if (length > buf_size) length = buf_size;
    int transferred = 0;
    int rc = ep_in_is_interrupt
        ? libusb_interrupt_transfer(usb_handle, ep_in, buf, length, &transferred, timeout_ms)
        : libusb_bulk_transfer(usb_handle, ep_in, buf, length, &transferred, timeout_ms);
    if (rc == LIBUSB_ERROR_TIMEOUT) return 0;
    if (rc != 0) return rc;
    return transferred;
}

static float read_f32_le(const uint8_t* buf) {
    float f;
    memcpy(&f, buf, sizeof(f));
    return f;
}

static uint32_t read_u32_le(const uint8_t* buf) {
    return (uint32_t) buf[0] | ((uint32_t) buf[1] << 8) | ((uint32_t) buf[2] << 16) | ((uint32_t) buf[3] << 24);
}

static void apply_device_info(const uint8_t* frame) {
    device_board_id = frame[DEVINFO_BOARD_ID_OFFSET];
    is_sbs_mode = frame[DEVINFO_SIDE_BY_SIDE_OFFSET] != 0;
    device_imu_rotation_x_deg = (device_board_id == BOARD_AIR_4_PRO) ? -20.0f : 0.0f;
}

// ---------------------------------------------------------------------------
// Pose ingestion
// ---------------------------------------------------------------------------
static uint32_t last_utilized_ts = 0;
static void handle_sensor_frame(const uint8_t* frame) {
    imu_vec3_type accel = {
        .x = read_f32_le(frame + SENSOR_ACCEL_OFFSET + 0),
        .y = read_f32_le(frame + SENSOR_ACCEL_OFFSET + 4),
        .z = read_f32_le(frame + SENSOR_ACCEL_OFFSET + 8)
    };
    imu_vec3_type gyro_dps = {
        .x = read_f32_le(frame + SENSOR_GYRO_OFFSET + 0),
        .y = read_f32_le(frame + SENSOR_GYRO_OFFSET + 4),
        .z = read_f32_le(frame + SENSOR_GYRO_OFFSET + 8)
    };
    uint32_t tick_100us = read_u32_le(frame + SENSOR_TICK_OFFSET);

    // always run the filter for accuracy, but throttle pose ingestion
    imu_quat_type filter_quat = filter_update(&filter, accel, gyro_dps, tick_100us);

    if (driver_disabled()) return;

    uint32_t ts = tick_100us / 10; // 100us units -> ms
    uint32_t elapsed = ts - last_utilized_ts;
    if (elapsed > FORCED_CYCLE_TIME_MS) {
        imu_quat_type nwu_quat = multiply_quaternions(filter_to_nwu_quat, filter_quat);

        imu_pose_type pose = (imu_pose_type){0};
        pose.orientation = nwu_quat;
        pose.has_orientation = true;
        pose.timestamp_ms = ts;
        connection_pool_ingest_pose(RAYNEO_DRIVER_ID, pose);

        last_utilized_ts = ts;
    }
}

// ---------------------------------------------------------------------------
// Device driver interface
// ---------------------------------------------------------------------------
bool rayneo_device_connect() {
    pthread_mutex_lock(&usb_mutex);
    if (!soft_connected) {
        if (!hard_connected) {
            hard_connected = rayneo_open_usb(0, 0);
        }
        if (hard_connected) {
            filter_reset(&filter, device_imu_rotation_x_deg);
            last_utilized_ts = 0;
            if (rayneo_send(CMD_OPEN_IMU, 0)) {
                soft_connected = true;
            } else {
                log_message("RayNeo driver, failed to start IMU stream\n");
            }
        } else {
            log_message("RayNeo driver, failed to establish a connection\n");
        }
    }
    pthread_mutex_unlock(&usb_mutex);

    return soft_connected;
}

void rayneo_device_disconnect(bool soft, bool is_device_present) {
    pthread_mutex_lock(&usb_mutex);
    if (soft_connected) {
        rayneo_send(CMD_CLOSE_IMU, 0);
        soft_connected = false;
    }

    bool retain_hard_connection = soft && is_device_present;
    if (hard_connected && !retain_hard_connection) {
        rayneo_close_usb();
        hard_connected = false;
    }
    pthread_mutex_unlock(&usb_mutex);
}

device_properties_type* rayneo_supported_device(uint16_t vendor_id, uint16_t product_id, uint8_t usb_bus, uint8_t usb_address) {
    if (vendor_id != RAYNEO_ID_VENDOR || product_id != RAYNEO_ID_PRODUCT) return NULL;

    device_properties_type* device = calloc(1, sizeof(device_properties_type));
    *device = rayneo_properties;
    device->usb_bus = usb_bus;
    device->usb_address = usb_address;

    // trying to connect to the device too quickly seems to cause connection issues
    sleep(2);

    // Best-effort: open early and request device info so we can report the correct model and pick
    // up the IMU mounting correction before the IMU stream begins. This must NOT gate detection -
    // if the open or handshake fails here, we still report the device as supported and let the
    // normal connect/retry path (and its logging) take over.
    pthread_mutex_lock(&usb_mutex);
    if (!hard_connected) hard_connected = rayneo_open_usb(usb_bus, usb_address);
    bool got_info = false;
    if (hard_connected && rayneo_send(CMD_ACQUIRE_DEVICE_INFO, 0)) {
        uint8_t buf[512];
        int64_t deadline = monotonic_ns() + 1500000000LL;
        while (!got_info && monotonic_ns() < deadline) {
            int n = rayneo_read_frame(buf, sizeof(buf), 200);
            if (n < 0) break;
            if (n >= FRAME_LEN && buf[0] == FRAME_IN_MAGIC && buf[1] == FRAME_TYPE_RESPONSE) {
                apply_device_info(buf);
                got_info = true;
            }
        }
    }
    pthread_mutex_unlock(&usb_mutex);

    if (got_info) {
        device->model = (device_board_id == BOARD_AIR_4_PRO) ? "Air 4 Pro" : "Air 3S Pro";
        if (config()->debug_device)
            log_debug("RayNeo driver, board id 0x%02X, sbs=%d\n", device_board_id, is_sbs_mode);
    } else {
        log_message("RayNeo driver, device detected but device info not yet read; will connect and retry\n");
    }

    // Leave the connection open if we think it'll be used, but if the driver is disabled, disconnect now
    if (driver_disabled()) rayneo_device_disconnect(true, true);

    return device;
}

void rayneo_block_on_device() {
    device_properties_type* device = device_checkout();
    if (device != NULL) rayneo_device_connect();

    uint8_t buf[512];
    int64_t last_info_request_ns = monotonic_ns();
    while (soft_connected && device != NULL) {
        pthread_mutex_lock(&usb_mutex);
        int n = soft_connected ? rayneo_read_frame(buf, sizeof(buf), 100) : LIBUSB_ERROR_NO_DEVICE;
        pthread_mutex_unlock(&usb_mutex);

        if (n < 0) {
            if (config()->debug_device) log_debug("RayNeo driver, read error %d, disconnecting\n", n);
            break;
        }

        if (n >= FRAME_LEN && buf[0] == FRAME_IN_MAGIC) {
            if (buf[1] == FRAME_TYPE_SENSOR) {
                handle_sensor_frame(buf);
            } else if (buf[1] == FRAME_TYPE_RESPONSE) {
                uint8_t prev_board = device_board_id;
                apply_device_info(buf);
                // if we only learned the board id now (not during detection), apply its IMU tilt
                if (device_board_id != prev_board)
                    filter.imu_rotation = imu_rotation_quat_from_deg(device_imu_rotation_x_deg);
            }
        }

        // periodically refresh device info to keep SBS state current
        int64_t now = monotonic_ns();
        if (now - last_info_request_ns > 1000000000LL) {
            pthread_mutex_lock(&usb_mutex);
            if (soft_connected) rayneo_send(CMD_ACQUIRE_DEVICE_INFO, 0);
            pthread_mutex_unlock(&usb_mutex);
            last_info_request_ns = now;
        }
    }

    rayneo_device_disconnect(true, device != NULL);
    device_checkin(device);
}

bool rayneo_device_is_sbs_mode() {
    return is_sbs_mode;
}

bool rayneo_device_set_sbs_mode(bool enabled) {
    pthread_mutex_lock(&usb_mutex);
    bool sent = rayneo_send(enabled ? CMD_SWITCH_TO_3D : CMD_SWITCH_TO_2D, 0);
    pthread_mutex_unlock(&usb_mutex);
    // the new state is confirmed when the next device info response comes back around
    return sent;
}

bool rayneo_is_connected() {
    return soft_connected;
}

void rayneo_disconnect(bool soft) {
    rayneo_device_disconnect(soft, device_present());
}

const device_driver_type rayneo_driver = {
    .id                                 = RAYNEO_DRIVER_ID,
    .supported_device_func              = rayneo_supported_device,
    .device_connect_func                = rayneo_device_connect,
    .block_on_device_func               = rayneo_block_on_device,
    .device_is_sbs_mode_func            = rayneo_device_is_sbs_mode,
    .device_set_sbs_mode_func           = rayneo_device_set_sbs_mode,
    .is_connected_func                  = rayneo_is_connected,
    .disconnect_func                    = rayneo_disconnect
};
