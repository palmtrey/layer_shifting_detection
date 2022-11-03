import os, time

# Takes inputs from OctoPrint and CNN
# Currently being set as constants until this is figured out
shift_detected = False
printCompletion = 0.0
printAction = "RUNNING..."

def display():
    os.system('cls||clear')
    if shift_detected:
        print("Shift detected.")
    else:
        print("No shift detected.")
    print("Print Completion: " + str(printCompletion) + "%")
    print("Print Action: " + printAction)
    print("\n\nCtrl+C to end")

while True:
    display()
    time.sleep(5)

    if printCompletion == 100:
        printAction = "Completed"
        break
    if shift_detected:
        printAction = "Paused"
        break
