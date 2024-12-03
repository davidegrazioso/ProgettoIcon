from apprendimentosupervisionato import leggendario
from Closest import closest

def main():
    while True:
        print("\nBenvenutə, Scegli cosa vuoi fare:")
        print("1. dimmi se un pokemon è leggendario o no")
        print("2. raccomandami dei pokemon")
        print("3. Esci")
        scelta = input("Inserisci il numero della tua scelta: ")

        if scelta == "1":
            leggendario()
        elif scelta == "2":
            closest()
        elif scelta == "3":
            print("Arrivederci!")
            break
        else:
            print("Scelta non valida. Riprova.")



if __name__ == "__main__":
    main()


