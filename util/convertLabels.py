# python script that converts CORe50 labels (.xtx) into .pbtxt for Tensorflow Object Detection
# input.txt --> CORe50 labels
# output.pbttx

input_file = open('input.txt', 'r').readlines()
output_file = open('output.pbtxt', 'w')
counter = 0


def addApex(s):
    newString = "'" + s + "'"
    return newString


def createItem(s):
    sf='item {\n id:' + str(counter)+' \n name: ' + repr(s) + '\n}\n\n'
    return sf


for i in input_file:
    counter+=1
    output_file.writelines(createItem(i.strip()))
output_file.close()

print('done! Your .pbtxt file is ready!')



