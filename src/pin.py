import getpass

def ask_pin():
    """
    Ask for a hardcoded pin code
    """
    print("#############################")
    print("Please enter the PIN code.")
    print("Entrez le code PIN svp.")
    print("Introduzca el PIN, por favor.")
    print("Tippen sie der PIN bitte.")
    print("#############################")
    print()
    while True:
        pswd = getpass.getpass(">")
        if pswd == "712556": break
        else: print("Wrong password, try again...\n")
    print("** PIN is correct! **\n")
    print("#############################")
    print("#############################\n")

if __name__ == '__main__': ask_pin()