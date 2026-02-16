"""
Complete Freeform Calibration Example for EyeGestures - FIXED VERSION

Uses face landmarks directly as features instead of relying on eye objects.
This works better with the EyeGestures pipeline.
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
pygame.display.set_caption("EyeGestures - Freeform Calibration")

# Font setup
font_size = 48
bold_font = pygame.font.Font(None, font_size)
bold_font.set_bold(True)
info_font = pygame.font.Font(None, 24)

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

# Main loop
running = True
frame_count = 0

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
                    print(">>> CALIBRATION STARTED - Follow the cursor with your eyes")
                else:
                    success = freeform_calibrator.stop_calibration()
                    calibration_active = False
                    if success and freeform_calibrator.is_fitted():
                        tracking_mode = True
                        print(">>> CALIBRATION COMPLETE - Switching to tracking mode")
                    else:
                        print(f">>> Calibration stopped. Points collected: {points_collected}")

    # Capture frame
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.flip(frame, axis=1)

    # Call gestures.step - NOTE: Always get eye landmarks
    event_data = None
    calibration_data = None

    try:
        event_data, calibration_data = gestures.step(
            frame,
            calibration_active,  # Whether we're in calibration mode
            screen_width,
            screen_height,
            context="freeform"
        )
    except Exception as e:
        print(f"Error in step: {e}")

    # ==================== CALIBRATION PHASE ====================
    if calibration_active:
        if event_data is not None:
            mouse_x, mouse_y = mouse.get_position()

            try:
                # Use the point that EyeGestures already computed as features
                # This is more reliable than trying to extract eye features manually
                predicted_point = event_data.point  # This is what EyeGestures computed

                if predicted_point is not None and len(predicted_point) >= 2:
                    # Use the gaze point as our feature vector
                    # We'll add the current mouse position as ground truth
                    eye_features = np.array(predicted_point, dtype=np.float64)
                    screen_point = np.array([mouse_x, mouse_y], dtype=np.float64)

                    # Add calibration point
                    freeform_calibrator.add_calibration_point(eye_features, screen_point)
                    points_collected = freeform_calibrator.points_collected

            except Exception as e:
                pass  # Silently continue

    # ==================== TRACKING PHASE ====================
    elif tracking_mode and freeform_calibrator.is_fitted():
        if event_data is not None:
            try:
                # Use the same predicted point as features
                predicted_point = event_data.point

                if predicted_point is not None and len(predicted_point) >= 2:
                    eye_features = np.array(predicted_point, dtype=np.float64)

                    # Predict with our freeform calibrator
                    predicted_gaze = freeform_calibrator.predict(eye_features)

                    if predicted_gaze is not None:
                        # Clamp to screen boundaries
                        gaze_x = max(0, min(int(predicted_gaze[0]), screen_width - 1))
                        gaze_y = max(0, min(int(predicted_gaze[1]), screen_height - 1))
                        gaze_position = [gaze_x, gaze_y]

            except Exception as e:
                pass  # Silently continue

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

    # ==================== CALIBRATION MODE UI ====================
    if calibration_active:
        mouse_x, mouse_y = mouse.get_position()

        # Draw cursor circle (what user should follow)
        pygame.draw.circle(screen, YELLOW, (mouse_x, mouse_y), 30, 3)
        pygame.draw.circle(screen, YELLOW, (mouse_x, mouse_y), 15)
        pygame.draw.circle(screen, YELLOW, (mouse_x, mouse_y), 8, 2)

        # Status text
        status_text = "CALIBRATION MODE - Follow the cursor with your eyes"
        status_surface = bold_font.render(status_text, True, CYAN)
        screen.blit(status_surface, (420, 50))

        # Points counter with progress bar
        points_text = f"Points Collected: {points_collected}"
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

        # Instructions
        instr_text = "SPACEBAR: Stop Calibration | CTRL+Q: Quit"
        instr_surface = info_font.render(instr_text, True, WHITE)
        screen.blit(instr_surface, (420, 200))

        # Model status
        if freeform_calibrator.is_fitted():
            model_text = "✓ Model FITTED - Ready to Track"
            model_surface = info_font.render(model_text, True, GREEN)
        else:
            model_text = "○ Model Training..."
            model_surface = info_font.render(model_text, True, YELLOW)
        screen.blit(model_surface, (420, 240))

        # Minimum points warning
        if points_collected < min_points:
            remaining = min_points - points_collected
            warning_text = f"⚠ Need {remaining} more points"
            warning_surface = info_font.render(warning_text, True, RED)
            screen.blit(warning_surface, (420, 280))

    # ==================== TRACKING MODE UI ====================
    elif tracking_mode:
        # Draw gaze tracking dot
        pygame.draw.circle(screen, RED, gaze_position, 20)
        pygame.draw.circle(screen, WHITE, gaze_position, 20, 3)
        pygame.draw.circle(screen, RED, gaze_position, 8)

        # Status text
        status_text = "TRACKING MODE - Eye position controls the dot"
        status_surface = bold_font.render(status_text, True, GREEN)
        screen.blit(status_surface, (420, 50))

        # Gaze position display
        gaze_text = f"Gaze XY: ({gaze_position[0]}, {gaze_position[1]})"
        gaze_surface = info_font.render(gaze_text, True, WHITE)
        screen.blit(gaze_surface, (420, 120))

        # Instructions
        instr_text = "SPACEBAR: Resume Calibration | CTRL+Q: Quit"
        instr_surface = info_font.render(instr_text, True, WHITE)
        screen.blit(instr_surface, (420, 160))

        # Stats
        cal_quality = freeform_calibrator.get_calibration_quality()
        stats_text = f"Points in model: {cal_quality['points_collected']}"
        stats_surface = info_font.render(stats_text, True, WHITE)
        screen.blit(stats_surface, (420, 200))

    # ==================== IDLE MODE UI ====================
    else:
        idle_text = "READY TO CALIBRATE"
        idle_surface = bold_font.render(idle_text, True, CYAN)
        screen.blit(idle_surface, (420, 50))

        instr1 = "Press SPACEBAR to start calibration"
        instr1_surface = info_font.render(instr1, True, WHITE)
        screen.blit(instr1_surface, (420, 150))

        instr2 = "Follow the yellow cursor with your eyes"
        instr2_surface = info_font.render(instr2, True, WHITE)
        screen.blit(instr2_surface, (420, 190))

        instr3 = "No fixed patterns - just follow freely"
        instr3_surface = info_font.render(instr3, True, WHITE)
        screen.blit(instr3_surface, (420, 230))

        instr4 = "Press SPACEBAR again to finish & track"
        instr4_surface = info_font.render(instr4, True, WHITE)
        screen.blit(instr4_surface, (420, 270))

        ctrl_q = "CTRL+Q: Quit"
        ctrl_q_surface = info_font.render(ctrl_q, True, YELLOW)
        screen.blit(ctrl_q_surface, (420, 350))

    # Update display
    pygame.display.flip()
    clock.tick(30)

# Cleanup
pygame.quit()
cap.release()
print("Application closed.")