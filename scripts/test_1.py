#!/usr/bin/env python
# coding: utf-8

import rospy
import cv2
import numpy as np

def image_process(img):

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    kernel_size = 3
    blur_img = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)

    low_threshold = 70
    high_threshold = 140
    canny_img = cv2.Canny(img, low_threshold, high_threshold)

    return canny_img

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        color = (255, 255, 255)
    else:
        color = 255

    cv2.fillPoly(mask, vertices, color)

    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image

def draw_lines(img, lines, color=[0, 0, 225], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength = min_line_len, maxLineGap = max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img):
    return cv2.addWeighted(initial_img, 1, img, 1, 0)

if __name__ == "__main__":
    rospy.init_node('image_test')

    cap = cv2.VideoCapture("/home/ej/test_video/yellow.mp4")

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        height, width = frame.shape[:2]

        canny_img = image_process(frame)
        vertices = np.array([[(50, height), (width/2-45, height/2+60), (width/2+45, height/2+60), (width-50, height)]], dtype=np.int32)
        ROI_img = region_of_interest(canny_img, vertices)
        hough_img = hough_lines(ROI_img, 1, 1*np.pi/180, 30, 10, 20)

        result = weighted_img(hough_img, frame)
        cv2.imshow('result', result), cv2.waitKey(1)

        if cv2.waitKey(3) & 0xFF == ord('q'):
            break

    cap.release()
