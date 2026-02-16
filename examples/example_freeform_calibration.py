"""
Complete Freeform Calibration Example for EyeGestures

User Controls:
- Press SPACEBAR to START/STOP calibration
- Move mouse cursor around the screen
- User follows the cursor with their eyes (no fixed patterns required)
- Calibration accumulates points automatically
- Once enough points are collected, the model learns in real-time
- User can use eye tracking to move a dot on screen
- If error increases, press SPACEBAR again to RESUME calibration from where it stopped
"""

import os
import sys
import cv2
import pygame
import numpy as np
import mouse

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

# Initialize EyeGestures and Freeform Calibrator
gestures = EyeGestures_v3()
freeform_calibrator = FreeformCalibrator(calibration_radius=1000)
cap = VideoCapture(0)

# Set up colors
RED = (255, 0, 100)
BLUE = (100, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)

# Initialize Pygame clock
clock = pygame.time.Clock()

# State variables
calibration_active = False
calibration_paused = False
points_collected = 0
tracking_mode = False  # True = using eye gaze to control dot, False = calibration mode
gaze_position = [screen_width // 2, screen_height // 2]
error_metric = 0
predicted_position = None

# Main game loop
running = True
frame_count = 0

while running:
    frame_count += 1

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            # CTRL+Q to quit
            if event.key == pygame.K_q and pygame.key.get_mods() & pygame.KMOD_CTRL:
                running = False

            # SPACEBAR to toggle calibration
            elif event.key == pygame.K_SPACE:
                if tracking_mode:
                    # Switch from tracking back to calibration
                    tracking_mode = False
                    freeform_calibrator.pause_calibration()
                    print("Switched to calibration mode - resuming calibration")
                else:
                    # Toggle calibration on/off
                    if not calibration_active:
                        freeform_calibrator.start_calibration()
                        calibration_active = True
                        calibration_paused = False
                        print(">>> CALIBRATION STARTED - Follow the cursor with your eyes")
                    else:
                        success = freeform_calibrator.stop_calibration()
                        calibration_active = False
                        if success and freeform_calibrator.is_fitted():
                            tracking_mode = True
                            print(">>> CALIBRATION COMPLETE - Switching to tracking mode")
                        else:
                            print(">>> Not enough points for calibration")

    # Capture frame from camera
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.flip(frame, axis=1)

    # Get eye tracking data from EyeGestures
    if calibration_active or tracking_mode:
        event_data, calibration_data = gestures.step(
            frame,
            False,  # Don't use EyeGestures' built-in calibration
            screen_width,
            screen_height,
            context="freeform"
        )
    else:
        event_data = None
        calibration_data = None

    # ==================== CALIBRATION PHASE ====================
    if calibration_active:
        if event_data is not None:
            # Get mouse position (this is the ground truth while calibrating)
            mouse_x, mouse_y = mouse.get_position()

            # Extract eye features from event_data
            # Using pupil position and landmark data as features
            try:
                if hasattr(event_data, 'l_eye') and event_data.l_eye:
                    left_pupil = event_data.l_eye.getPupil()
                    left_landmarks = event_data.l_eye.getLandmarks()
                    left_center = event_data.l_eye.getCenter()

                    if hasattr(event_data, 'r_eye') and event_data.r_eye:
                        right_pupil = event_data.r_eye.getPupil()
                        right_landmarks = event_data.r_eye.getLandmarks()
                        right_center = event_data.r_eye.getCenter()

                        # Combine left and right eye features into a single feature vector
                        if left_pupil is not None and right_pupil is not None:
                            eye_features = np.concatenate([
                                left_pupil.flatten(),
                                right_pupil.flatten(),
                                [left_center[0], left_center[1]],
                                [right_center[0], right_center[1]]
                            ])

                            # Add calibration point
                            screen_point = np.array([mouse_x, mouse_y])
                            freeform_calibrator.add_calibration_point(eye_features, screen_point)
                            points_collected = freeform_calibrator.points_collected
            except Exception as e:
                pass  # Silently continue if feature extraction fails

    # ==================== TRACKING PHASE ====================
    elif tracking_mode and freeform_calibrator.is_fitted():
        if event_data is not None:
            try:
                if hasattr(event_data, 'l_eye') and event_data.l_eye:
                    left_pupil = event_data.l_eye.getPupil()
                    right_pupil = event_data.r_eye.getPupil()
                    left_center = event_data.l_eye.getCenter()
                    right_center = event_data.r_eye.getCenter()

                    if left_pupil is not None and right_pupil is not None:
                        # Create feature vector
                        eye_features = np.concatenate([
                            left_pupil.flatten(),
                            right_pupil.flatten(),
                            [left_center[0], left_center[1]],
                            [right_center[0], right_center[1]]
                        ])

                        # Predict gaze position
                        predicted_position = freeform_calibrator.predict(eye_features)

                        if predicted_position is not None:
                            gaze_position = [int(predicted_position[0]), int(predicted_position[1])]
            except Exception as e:
                pass  # Silently continue if prediction fails

    # ==================== RENDERING ====================
    screen.fill(BLACK)

    # Display camera frame
    if event_data and hasattr(event_data, 'sub_frame') and event_data.sub_frame is not None:
        frame_surface = pygame.surfarray.make_surface(np.rot90(event_data.sub_frame))
        frame_surface = pygame.transform.scale(frame_surface, (400, 400))
        screen.blit(frame_surface, (0, 0))

    # ==================== CALIBRATION MODE UI ====================
    if calibration_active:
        # Get mouse position for display
        mouse_x, mouse_y = mouse.get_position()

        # Draw cursor circle (what user should follow)
        pygame.draw.circle(screen, YELLOW, (mouse_x, mouse_y), 25, 3)
        pygame.draw.circle(screen, YELLOW, (mouse_x, mouse_y), 10)

        # Display calibration info
        cal_quality = freeform_calibrator.get_calibration_quality()

        # Status text
        status_text = "CALIBRATION MODE - Follow the cursor with your eyes"
        status_surface = bold_font.render(status_text, True, CYAN)
        screen.blit(status_surface, (420, 50))

        # Points counter
        points_text = f"Points Collected: {points_collected}"
        points_surface = bold_font.render(points_text, True, GREEN)
        screen.blit(points_surface, (420, 120))

        # Instructions
        instr_text = "SPACEBAR: Stop Calibration"
        instr_surface = info_font.render(instr_text, True, WHITE)
        screen.blit(instr_surface, (420, 200))

        # Model status
        if freeform_calibrator.is_fitted():
            model_text = "Model: FITTED ✓"
            model_surface = info_font.render(model_text, True, GREEN)
        else:
            model_text = "Model: Training..."
            model_surface = info_font.render(model_text, True, YELLOW)
        screen.blit(model_surface, (420, 240))

        # Minimum points warning
        if points_collected < 10:
            warning_text = f"Need {10 - points_collected} more points for calibration"
            warning_surface = info_font.render(warning_text, True, RED)
            screen.blit(warning_surface, (420, 280))

    # ==================== TRACKING MODE UI ====================
    elif tracking_mode:
        # Draw gaze tracking dot
        pygame.draw.circle(screen, RED, gaze_position, 15)
        pygame.draw.circle(screen, WHITE, gaze_position, 15, 2)

        # Status text
        status_text = "TRACKING MODE - Eye position controls the dot"
        status_surface = bold_font.render(status_text, True, GREEN)
        screen.blit(status_surface, (420, 50))

        # Gaze position display
        gaze_text = f"Gaze: ({gaze_position[0]}, {gaze_position[1]})"
        gaze_surface = info_font.render(gaze_text, True, WHITE)
        screen.blit(gaze_surface, (420, 120))

        # Instructions
        instr_text = "SPACEBAR: Resume Calibration"
        instr_surface = info_font.render(instr_text, True, WHITE)
        screen.blit(instr_surface, (420, 200))

        # Stats
        cal_quality = freeform_calibrator.get_calibration_quality()
        stats_text = f"Points in model: {cal_quality['points_collected']}"
        stats_surface = info_font.render(stats_text, True, WHITE)
        screen.blit(stats_surface, (420, 240))

    # ==================== IDLE MODE UI ====================
    else:
        # Initial screen
        idle_text = "READY TO CALIBRATE"
        idle_surface = bold_font.render(idle_text, True, CYAN)
        screen.blit(idle_surface, (420, 50))

        instr1 = "Press SPACEBAR to start calibration"
        instr1_surface = info_font.render(instr1, True, WHITE)
        screen.blit(instr1_surface, (420, 150))

        instr2 = "Follow the cursor with your eyes"
        instr2_surface = info_font.render(instr2, True, WHITE)
        screen.blit(instr2_surface, (420, 190))

        instr3 = "No fixed patterns required"
        instr3_surface = info_font.render(instr3, True, WHITE)
        screen.blit(instr3_surface, (420, 230))

        instr4 = "Press SPACEBAR again to finish"
        instr4_surface = info_font.render(instr4, True, WHITE)
        screen.blit(instr4_surface, (420, 270))

        ctrl_q = "CTRL+Q: Quit"
        ctrl_q_surface = info_font.render(ctrl_q, True, YELLOW)
        screen.blit(ctrl_q_surface, (420, 350))

    # Update display
    pygame.display.flip()
    clock.tick(60)

# Cleanup
pygame.quit()
cap.release()
print("Application closed.")