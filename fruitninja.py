import cv2
import mediapipe as mp
import random

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)



# Initialize the webcam
cap = cv2.VideoCapture(0)

# Game variables
score = 0
misses = 0
game_over = False

# Load fruit images
fruit_images = {
    "apple": cv2.imread('apple_small.png'),
    "banana": cv2.imread('banana_small.png'),
    "orange": cv2.imread('orange_small.png')
}

# Resize fruit images
for fruit_name, fruit_image in fruit_images.items():
    fruit_images[fruit_name] = cv2.resize(fruit_image, (64, 64))  # Adjust the size as needed

# Function to spawn a fruit
def spawn_fruit():
    fruit_type = random.choice(list(fruit_images.keys()))
    x = random.randint(50, 600)  # Random x-coordinate within the frame width
    y = 0
    speed = 5  # Adjusted speed for the banana falling
    return {"type": fruit_type, "image": fruit_images[fruit_type], "x": x, "y": y, "speed": speed}

# Fruits on screen
fruits_on_screen = []

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    if not ret:
        break

    # Detect hand keypoints
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    #x,y = None , None
    

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        hand_landmarks = hand_landmarks.landmark[8]  # Select the landmark for the tip of the index finger
        x = int(hand_landmarks.x * frame.shape[1])  # Convert the x-coordinate to the frame's width
        y = int(hand_landmarks.y * frame.shape[0])  # Convert the y-coordinate to the frame's height
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        #landmark_positions.append((x,y))

        # Check collisions with fruits
        for fruit in fruits_on_screen:
            fruit_width = fruit["image"].shape[1]
            fruit_height = fruit["image"].shape[0]
            if x > fruit["x"] and x < fruit["x"] + fruit_width and y > fruit["y"] and y < fruit["y"] + fruit_height:
                score += 1
                fruits_on_screen.remove(fruit)

    # Drop fruits
    if random.random() < 0.1 and not game_over:  # Adjusted frequency of dropping fruits
        fruits_on_screen.append(spawn_fruit())

    # Update fruit positions
    for fruit in fruits_on_screen:
        fruit["y"] += fruit["speed"]
        if fruit["y"] > frame.shape[0]:
            misses += 1
            fruits_on_screen.remove(fruit)

    # Display fruits
    for fruit in fruits_on_screen:
        frame[fruit["y"]:fruit["y"]+fruit["image"].shape[0], fruit["x"]:fruit["x"]+fruit["image"].shape[1]] = fruit["image"]

    # Display score and misses
    cv2.putText(frame, f"Score: {score}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Misses: {misses}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display game over message
    if game_over:
        cv2.putText(frame, "Game Over!", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    cv2.imshow("Fruit Ninja", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
