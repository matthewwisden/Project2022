import cv2 as cv
import numpy as np

# Distance variables
KNOWN_DISTANCE = 114.3  # CM
PLAYER_WIDTH = 27  # CM
BALL_WIDTH = 10  # CM
GOAL_WIDTH = 50  # CM
# Object detection variables
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
# defining fonts
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv.dnn.readNet('yolov4_training_last.weights', 'yolov4_testing.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)



def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):

        color = COLORS[int(classid) % len(COLORS)]

        label = "%s : %f" % (class_names[classid[0]], score)

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1] - 14), FONTS, 0.5, color, 2)


        if classid == 0:  # Player class id
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1] - 2)])
        elif classid == 1:  # Ball
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1] - 2)])
        elif classid == 2:  # car
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1] - 2)])
    return data_list


def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length


# distance finder function
def distance_finder(focal_length, real_object_width, width_in_frame):
    distance = (real_object_width * focal_length) / width_in_frame
    return distance


# reading the reference image from dir
ref_player = cv.imread('player.jpg')
ref_ball = cv.imread('ball.jpg')
ref_goal = cv.imread('goal.jpg')

##goal_data = object_detector(ref_goal)
##goal_width_in_rf = goal_data[2][1]
player_data = object_detector(ref_player)
player_width_in_rf = player_data[0][1]

ball_data = object_detector(ref_ball)
ball_width_in_rf = ball_data[1][1]

goal_data = object_detector(ref_goal)
goal_width_in_rf = goal_data[2][1]


print(f"player width in pixels : {player_width_in_rf} ball width in pixel: {ball_width_in_rf} ball width in pixel: {goal_width_in_rf}")

# finding focal length
focal_player = focal_length_finder(KNOWN_DISTANCE, PLAYER_WIDTH, player_width_in_rf)

focal_ball = focal_length_finder(KNOWN_DISTANCE, BALL_WIDTH, ball_width_in_rf)

focal_goal = focal_length_finder(KNOWN_DISTANCE, GOAL_WIDTH, goal_width_in_rf)
cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()

    data = object_detector(frame)
    for d in data:
        if d[0] == 'player':
            distance = distance_finder(focal_player, PLAYER_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'ball':
            distance = distance_finder(focal_ball, BALL_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'goal':
            distance = distance_finder(focal_goal, GOAL_WIDTH, d[1])
            x, y = d[2]
        cv.rectangle(frame, (x, y - 3), (x + 150, y + 23), BLACK, -1)
        cv.putText(frame, f'Dis: {round(distance, 2)} inch', (x + 5, y + 13), FONTS, 0.48, GREEN, 2)

    cv.imshow('frame', frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break
cv.destroyAllWindows()
cap.release()