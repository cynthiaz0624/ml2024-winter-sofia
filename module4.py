def getNumbers():
    #The program asks the user for input N (positive integer) and reads it
    
    num = input("Enter a postive number")
    return int(num)
    
def getNumberOneByOne():
    #the program asks the user to provide N numbers (one by one) and reads all of them (again, one by one)
    
    N = input("Enter a number(positive): ")
    for i in range(1, N + 1):
      num = int(input("Enter a number: "))
      print("Number {i} is: {num}")  
    

def getIndex():
    X = input()
    numbers = getNumberOneByOne()
    if X not in numbers:
        return -1
    for i in numbers:
        if X == numbers[i]:
            return i
