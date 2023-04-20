def letters_to_number(letters):
    letters = letters.lower()
    num = 0
    for i in range(len(letters)):
        power = len(letters)-i-1
        num += (ord(letters[i]) - ord('a') + 1)*(26**power)
    return num


def number_to_letters(number):
    pass

option = 1#int(input("Do you want to convert letters to number (1) or number to letter(2): ").strip())

if option == 1:
    letters = input("Input column letters: ")
    print(letters_to_number(letters))
if option == 2:
   # letters = input("Input column number: ")
   print("Sorry")