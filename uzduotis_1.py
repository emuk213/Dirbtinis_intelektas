import numpy as np
import matplotlib.pyplot as plt
import math

"""
Ši funkcija sugeneruoja duomenis pasirinktuose intervaluose, juos atspausdina ir
sudeda į bendrą sąrašą. Pirmiausia sąraše eina mėlyni taškai, vėliau raudoni.
Funkcija  taip pat pavaizduoja duomenis Dekarto koordinačių sistemoje
ir grąžina visų taškų sąrašą, atskirus x ir y sąrašus.
"""
def tasku_grupes():

    ## Nustatoma, kad kiekvieną kartą generuojant kodą būtų gaunamos tos pačios reikšmės
    np.random.seed(47)

    ## Generuojami atsitiktiniai duomenys
    m_x = np.random.uniform(-1, 0.8, 10).tolist()
    m_y = np.random.uniform(-1, 0.8, 10).tolist()

    r_x = np.random.uniform(1, 2.8, 10).tolist()
    r_y = np.random.uniform(1, 2.8, 10).tolist()

    ## x ir y sujungiami į poras ir sudedami į mėlynų ir raudonų taškų sąrašus
    taskai_m = list(zip(m_x, m_y))
    taskai_r = list(zip(r_x, r_y))

    m = np.column_stack((m_x, m_y))
    r = np.column_stack((r_x, r_y))

    ## Sujungiame grupes į vieną
    taskai_visi = np.vstack((m, r))

    ## Duomenų grafikas
    plt.scatter(m_x, m_y, color="blue", label="Mėlyni")
    plt.scatter(r_x, r_y, color="red", label="Raudoni")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Tiesiškai atskiriami taškai")
    plt.grid(True)
    plt.show()

    ## Atspausdinami duomenys
    print(f'Mėlyni taškai:\n{taskai_m} \nRaudoni taškai:\n{taskai_r}')

    return taskai_visi, r_x, r_y, m_x, m_y

## Duomenų generavimo funkcija iškviečiama
taskai_visi, r_x, r_y, m_x, m_y  = tasku_grupes()

"""
Funkcija pagal svorių ir poslinkių rinkinius nubraižo tieses, skiriančias klases,
ir vektorius, atitinkančius gautus neurono svorius.
"""
def tieses(r_x, r_y, m_x, m_y, wwb):

    w1 = wwb[0][0]
    w2 = wwb[0][1]
    b = wwb[0][2]

    w12 = wwb[1][0]
    w22 = wwb[1][1]
    b2 = wwb[1][2]

    w13 = wwb[2][0]
    w23 = wwb[2][1]
    b3 = wwb[2][2]

    ## Parenkami x tiesės braižymui
    x = np.linspace(-1.5, 3, 20)

    ## Pagal svorius ir poslinkius apskaičiuojami tiesių y trims svorių ir poslinkių rinkiniams
    ## Formulė išvesta iš a reikšmės formulės
    y1 = (-(x * w1) - b) / w2
    y2 = (-(x * w12) - b2) / w22
    y3 = (-(x * w13) - b3) / w23

    ## Apskaičiuojama vektoriaus pradžia (ant tiesės)
    x0 = 1.5
    y01 = (-(x0 * w1) - b) / w2
    y02 = (-(x0 * w12) - b2) / w22
    y03 = (-(x0 * w13) - b3) / w23

    ## Braižomas grafikas su tiesėmis ir vektoriais
    plt.scatter(m_x, m_y, color="blue", label="Mėlyni(0)")
    plt.scatter(r_x, r_y, color="red", label="Raudoni(1)")
    plt.plot(x, y1, color="pink", linewidth=2, label="1-o rinkinio skiriamasis paviršius")
    plt.plot(x, y2, color="orange", linewidth=2, label="2-o rinkinio skiriamasis paviršius")
    plt.plot(x, y3, color="green", linewidth=2, label="3-o rinkinio skiriamasis paviršius")
    plt.quiver(x0, y01, w1, w2, angles='xy', scale_units='xy', scale=1, color='pink', label="1-o rinkinio svorių vektorius")
    plt.quiver(x0, y02, w12, w22, angles='xy', scale_units='xy', scale=1, color='orange', label="2-o rinkinio svorių vektorius")
    plt.quiver(x0, y03, w13, w23, angles='xy', scale_units='xy', scale=1, color='green', label="3-o rinkinio svorių vektorius")
    plt.legend(fontsize='small', loc="lower left")
    plt.axis("equal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Duomenys, svorių vektoriai ir tiesės, atskiriančios klases")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


## Suteikiama galimybė vartotojui pasirinkti aktyvacijos funkcijos tipą
print('Pasirinkite funkciją (slenkstinė - 0, sigmoidinė - 1)')
tipas = int(input())

"""
Funkcija realizuoja dirbtinį neuroną. Sukuriamas sąrašas su duomenų klasių reikšmėmis:
mėlyni taškai - 0, raudoni - 1. Atsitiktinai generuojant, vartotojo pasirinktai
funkcijai, randami tinkami trys svorių ir poslinkių rinkiniai.
Funkcija grąžina šiuos rinkinius ir vykdo klasifikaciją.
"""
def neuronas(taskai_visi, tipas):
    f_reiksmes = []
    ats = []
    taskai_visi = taskai_visi.tolist()

    ## Norimos reikšmės, mėlyni taškai 0, raudoni 1
    t = [0] * 10 + [1] * 10
    while len(ats) != 3:
        wwb = np.random.uniform(-1, 1, 3).tolist()
        f_reiksmes.clear()
        for i in range(20):
            a = wwb[0] * taskai_visi[i][0] + wwb[1] * taskai_visi[i][1] + wwb[2]

            ## Slenkstinė aktyvacijos funkcija
            if tipas == 0:
                if a >= 0:
                    f = 1
                else:
                    f = 0

            ## Sigmoidinė aktyvacijos funkcija
            if tipas == 1:
                f = 1/(1 + math.exp(-a))
                f = round(f)

            f_reiksmes.append(f)

        if f_reiksmes == t:
            ats.append(wwb)

    return ats


## Iškviečiama neurono funkcija
wwb = neuronas(taskai_visi, tipas)

## Atspausdinami rezultatai
if tipas == 0:
    ## Iškviečiama tiesių ir vektorių grafiko funkcija
    tieses(r_x, r_y, m_x, m_y, wwb)
    print(f'Slenkstinės funkcijos svorių ir poslinkių rinkinys:\n{wwb}')
else:
    print(f'Sigmoidinės funkcijos svorių ir poslinkių rinkinys:\n{wwb}')
