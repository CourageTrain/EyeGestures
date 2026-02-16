"""
Freeform Calibration - DEBUG VERSION
Shows what's happening during calibration
"""

import os
import sys
import cv2
import pygame
import numpy as np
import mouse
import time

pygame.init()
pygame.font.init()

# Get the display dimensions
screen_info = pygame.display.Info()
screen_width = screen_info.current_w
screen_height = screen_info.current_h

# Set up the screen
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("EyeGestures - Freeform Calibration (DEBUG)")

# Font setup
font_size = 48
bold_font = pygame.font.Font(None, font_size)
bold_font.set_bold(True)
info_font = pygame.font.Font(None, 18)
debug_font = pygame.font.Font(None, 16)

# Add parent directory to path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{dir_path}/..')

from eyeGestures.utils import VideoCapture
from eyeGestures import EyeGestures_v3
from eyeGestures.calibration_freeform import FreeformCalibrator

# Initialize
gestures = EyeGestures_v3()
freeform_calibrator = FreeformCalibrator(calibration_radius=1000)
cap = VideoCapture(0)

# Colors
RED = (255, 0, 100)
BLUE = (100, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)

clock = pygame.time.Clock()

# State
calibration_active = False
tracking_mode = False
gaze_position = [screen_width // 2, screen_height // 2]
points_collected = 0

# DEBUG INFO
debug_logs = []
frame_count = 0
event_data_count = 0
features_extracted_count = 0


def add_debug_log(msg):
    global debug_logs
    debug_logs.append(msg)
    if len(debug_logs) > 15:
        debug_logs.pop(0)
    print(f"[DEBUG] {msg}")


# Main loop
running = True

while running:
    frame_count += 1

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q and pygame.key.get_mods() & pygame.KMOD_CTRL:
                running = False
            elif event.key == pygame.K_SPACE:
                if not calibration_active:
                    freeform_calibrator.start_calibration()
                    calibration_active = True
                    tracking_mode = False
                    debug_logs.clear()
                    event_data_count = 0
                    features_extracted_count = 0
                    add_debug_log("CALIBRATION STARTED")
                else:
                    success = freeform_calibrator.stop_calibration()
                    calibration_active = False
                    add_debug_log(f"CALIBRATION STOPPED - Success: {success}")
                    if success and freeform_calibrator.is_fitted():
                        tracking_mode = True
                        add_debug_log("Switching to TRACKING MODE")

    # Capture frame
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.flip(frame, axis=1)

    # Call gestures.step
    event_data = None
    calibration_data = None

    try:
        event_data, calibration_data = gestures.step(
            frame,
            calibration_active,
            screen_width,
            screen_height,
            context="freeform"
        )

        if event_data is not None:
            event_data_count += 1

    except Exception as e:
        add_debug_log(f"ERROR in step(): {str(e)[:50]}")

    # DEBUG: Check what we got
    if frame_count % 30 == 0:  # Log every 30 frames
        if event_data is None:
            add_debug_log(f"event_data is None (frame {frame_count})")
        else:
            add_debug_log(f"event_data received! Has l_eye: {hasattr(event_data, 'l_eye')}")

    # ==================== CALIBRATION PHASE ====================
    if calibration_active:
        if event_data is not None:
            mouse_x, mouse_y = mouse.get_position()

            try:
                # Check attributes
                has_l_eye = hasattr(event_data, 'l_eye')
                has_r_eye = hasattr(event_data, 'r_eye')

                if not has_l_eye or not has_r_eye:
                    if frame_count % 60 == 0:
                        add_debug_log(f"Missing eyes: l_eye={has_l_eye}, r_eye={has_r_eye}")

                else:
                    l_eye = event_data.l_eye
                    r_eye = event_data.r_eye

                    if l_eye is None or r_eye is None:
                        if frame_count % 60 == 0:
                            add_debug_log(f"Eyes are None: l_eye={l_eye is not None}, r_eye={r_eye is not None}")
                    else:
                        left_pupil = l_eye.getPupil()
                        left_center = l_eye.getCenter()
                        right_pupil = r_eye.getPupil()
                        right_center = r_eye.getCenter()

                        if left_pupil is None or right_pupil is None:
                            if frame_count % 60 == 0:
                                add_debug_log(
                                    f"Pupils are None: l_pupil={left_pupil is not None}, r_pupil={right_pupil is not None}")
                        else:
                            # Create feature vector
                            eye_features = np.concatenate([
                                left_pupil.flatten(),
                                right_pupil.flatten(),
                                [left_center[0], left_center[1]],
                                [right_center[0], right_center[1]]
                            ])

                            # Add point
                            screen_point = np.array([mouse_x, mouse_y])
                            freeform_calibrator.add_calibration_point(eye_features, screen_point)
                            points_collected = freeform_calibrator.points_collected
                            features_extracted_count += 1

                            if features_extracted_count % 10 == 0:
                                add_debug_log(f"Points added: {points_collected}")

            except Exception as e:
                add_debug_log(f"Feature extraction error: {str(e)[:50]}")

    # ==================== RENDERING ====================
    screen.fill(BLACK)

    # Display camera frame
    if event_data and hasattr(event_data, 'sub_frame') and event_data.sub_frame is not None:
        try:
            frame_surface = pygame.surfarray.make_surface(np.rot90(event_data.sub_frame))
            frame_surface = pygame.transform.scale(frame_surface, (400, 400))
            screen.blit(frame_surface, (0, 0))
        except:
            pass

    # ==================== STATUS DISPLAY ====================
    if calibration_active:
        mouse_x, mouse_y = mouse.get_position()
        pygame.draw.circle(screen, YELLOW, (mouse_x, mouse_y), 30, 3)
        pygame.draw.circle(screen, YELLOW, (mouse_x, mouse_y), 15)

        status_text = "CALIBRATION MODE"
        status_surface = bold_font.render(status_text, True, CYAN)
        screen.blit(status_surface, (420, 50))

        points_text = f"Points: {points_collected}"
        points_surface = bold_font.render(points_text, True, GREEN)
        screen.blit(points_surface, (420, 120))

        # Progress bar
        min_points = 10
        progress = min(points_collected / min_points, 1.0)
        bar_width = 300
        bar_height = 20
        bar_x = 420
        bar_y = 160
        pygame.draw.rect(screen, WHITE, (bar_x, bar_y, bar_width, bar_height), 2)
        pygame.draw.rect(screen, GREEN, (bar_x, bar_y, int(bar_width * progress), bar_height))

        # Stats
        stats_text = f"event_data received: {event_data_count} frames"
        stats_surface = info_font.render(stats_text, True, WHITE)
        screen.blit(stats_surface, (420, 200))

        stats_text2 = f"Features extracted: {features_extracted_count}"
        stats_surface2 = info_font.render(stats_text2, True, WHITE)
        screen.blit(stats_surface2, (420, 225))

        instr_text = "SPACEBAR: Stop | CTRL+Q: Quit"
        instr_surface = info_font.render(instr_text, True, WHITE)
        screen.blit(instr_surface, (420, 260))

    else:
        idle_text = "READY - Press SPACEBAR to calibrate"
        idle_surface = bold_font.render(idle_text, True, CYAN)
        screen.blit(idle_surface, (420, 50))

    # ==================== DEBUG LOG DISPLAY ====================
    debug_y = screen_height - len(debug_logs) * 20 - 20
    for i, log in enumerate(debug_logs):
        log_surface = debug_font.render(log, True, YELLOW)
        screen.blit(log_surface, (420, debug_y + i * 20))

    # FPS counter
    fps = int(clock.get_fps())
    fps_text = f"FPS: {fps} | Frame: {frame_count}"
    fps_surface = debug_font.render(fps_text, True, WHITE)
    screen.blit(fps_surface, (420, screen_height - 20))

    # Update
    pygame.display.flip()
    clock.tick(30)

# Cleanup
pygame.quit()
cap.release()
print("Application closed.")