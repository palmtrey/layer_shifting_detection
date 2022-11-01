import os, time

# Takes inputs from OctoPrint and CNN
# Currently being set as constants until this is figured out
numShiftsDetected = 0
printCompletion = 0.0
printAction = "RUNNING..."

def display():
    os.system('cls||clear')
    print("Status: " + str(numShiftsDetected) + " Shifts Detected")
    print("Print Completion: " + str(printCompletion) + "%")
    print("Print Action: " + printAction)
    print("\n\nCtrl+C to end")

while True:
    display()
    time.sleep(5)

    if printCompletion == 100:
        printAction = "Completed"
        break
    if numShiftsDetected > 0:
        printAction = "Paused"
        break
