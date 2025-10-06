import pandas as pd
import math
import numpy as np
import time
import matplotlib.pyplot as plt

# --- DUOMENŲ PARUOŠIMAS ---
df = pd.read_csv("breast-cancer-wisconsin.data", sep=",", header=None)
# Pašalinamos trūkstamos reikšmės
df = df.replace("?", pd.NA)
df = df.dropna()
df = df.astype(int)

df.columns = ['Sample_code_number', 'Clump_thickness', 'Uniformity_of_cell_size',
              'Uniformity_of_cell_shape', 'Marginal_adhesion', 'Single_epithelial_cell_size',
              'Bare_nuclei', 'Bland_chromatin', 'Normal_nucleoli', 'Mitoses', 'Class']
df['Class'] = (df['Class'] == 4).astype(int) # Pakeičiamos klasių reikšmės į 0 ir 1 (2 -> 0, 4 -> 1)
df = df.drop(['Sample_code_number'], axis=1)  # Pašalinamas ID stulpelis

df.insert(0, 0, 1)  # Pridedamas stulpelis iš 1 (poslinkio radimui)

df = df.sample(frac=1, random_state=46)  # Duomenys sumaišomi

# Padalinama į mokymo, validavimo ir testavimo aibes
learn = df.sample(frac=0.8, replace=False, random_state=1)
rest_df = df.drop(learn.index)
test = rest_df.sample(frac=0.5, replace=False, random_state=1)
validate = rest_df.drop(test.index)

# Sukuriame X (požymiai) ir y (klasė) vektorius
X_learn = learn.iloc[:, :-1].to_numpy()
y_learn = learn.iloc[:, -1].to_numpy()

X_validate = validate.iloc[:, :-1].to_numpy()
y_validate = validate.iloc[:, -1].to_numpy()

X_test = test.iloc[:, :-1].to_numpy()
y_test = test.iloc[:, -1].to_numpy()

# --- FUNKCIJOS ---
def sigmoidine(a):
    return 1 / (1 + math.exp(-a))


def galutine_paklaida(X, y, w):
    totalError = 0
    for i in range(len(X)):
        a_i = sum(w[j] * X[i, j] for j in range(X.shape[1]))
        y_i = sigmoidine(a_i)  # Išėjimo reikšmė
        error_i = (y[i] - y_i) ** 2
        totalError += error_i  # Paklaidos sudedamos
    return totalError / len(X)  # Vidutinė paklaida


def tikslumas(X, y, w):
    teisingai = 0
    for i in range(len(X)):
        a = sum(w[j] * X[i, j] for j in range(X.shape[1]))
        f = round(sigmoidine(a))  # Apskaičiuojama gauta klasė
        if f == y[i]:
            teisingai += 1
    return teisingai / len(X)


# --- PAKETINIS ---
def paketinis(X, y, epochs, speed, X_val, y_val):
    totalError = math.inf
    epoch = 0
    paklaidos_po_epochos_l = []
    paklaidos_po_epochos_v = []
    tikslumas_po_epochos_l = []
    tikslumas_po_epochos_v = []
    Emin = 0.01
    # Generuojami atsitiktiniai svoriai (w_0 = b):
    np.random.seed(46)
    w = np.random.uniform(-1, 1, X.shape[1]).tolist()

    while (totalError > Emin) and (epoch < epochs):
        totalError = 0
        gradientSum = [0] * X.shape[1]

        for i in range(len(X)):
            t_i = y[i]  # Tikroji klasė (iš duomenų)
            a = sum(w[j] * X[i, j] for j in range(X.shape[1]))
            y_i = sigmoidine(a) # Gauta išėjimo reikšmė
          # Skaičiuojama gradiento suma:
            for k in range(X.shape[1]):
                gradientSum[k] += (y_i - t_i) * y_i * (1 - y_i) * X[i, k]
            error = (t_i - y_i) ** 2
            totalError += error
        # Svorių atnaujinimas:
        for k in range(X.shape[1]):
            w[k] = w[k] - speed * (gradientSum[k] / len(X))  # Gradiento vidurkis
          
        # Kaupiami tikslumai ir paklaidos:
        tikslumas_po_epochos_l.append(tikslumas(X, y, w))
        tikslumas_po_epochos_v.append(tikslumas(X_val, y_val, w))
        paklaidos_po_epochos_l.append(galutine_paklaida(X, y, w))
        paklaidos_po_epochos_v.append(galutine_paklaida(X_val, y_val, w))
        epoch += 1

    return {
        "w": w,
        "mokymo_paklaidos": paklaidos_po_epochos_l,
        "validavimo_paklaidos": paklaidos_po_epochos_v,
        "mokymo_tikslumai": tikslumas_po_epochos_l,
        "validavimo_tikslumai": tikslumas_po_epochos_v
    }


# --- STOCHASTINIS ---
def stochastinis(X, y, epochs, speed, X_val, y_val):
    totalError = math.inf
    paklaidos_po_epochos_l = []
    paklaidos_po_epochos_v = []
    tikslumas_po_epochos_l = []
    tikslumas_po_epochos_v = []
    epoch = 0
    Emin = 0.01
    # Generuojami atsitiktiniai svoriai:
    np.random.seed(46)
    w = np.random.uniform(-1, 1, X.shape[1]).tolist()

    while (totalError > Emin) and (epoch < epochs):
        totalError = 0
        for i in range(len(X)):
            t_i = y[i]
            a = sum(w[j] * X[i, j] for j in range(X.shape[1]))
            y_i = sigmoidine(a)
          # Svorių atnaujinimas:
            for k in range(X.shape[1]):
                w[k] = w[k] - speed * (y_i - t_i) * y_i * (1 - y_i) * X[i, k]
            error = (t_i - y_i) ** 2
            totalError += error
          
        # Kaupiami tikslumai ir paklaidos:
        paklaidos_po_epochos_l.append(galutine_paklaida(X, y, w))
        tikslumas_po_epochos_l.append(tikslumas(X, y, w))
        tikslumas_po_epochos_v.append(tikslumas(X_val, y_val, w))
        paklaidos_po_epochos_v.append(galutine_paklaida(X_val, y_val, w))
        epoch += 1

    return {
        "w": w,
        "mokymo_paklaidos": paklaidos_po_epochos_l,
        "validavimo_paklaidos": paklaidos_po_epochos_v,
        "mokymo_tikslumai": tikslumas_po_epochos_l,
        "validavimo_tikslumai": tikslumas_po_epochos_v
    }


# --- GRAFIKAI ---
def plot_paklaidos(paklaidos_l, paklaidos_v, method, speed, ep):
    plt.plot(range(1, len(paklaidos_l)+1), paklaidos_l, label="Mokymas")
    plt.plot(range(1, len(paklaidos_v)+1), paklaidos_v, label="Validavimas")
    plt.title(f"Paklaidų priklausomybė nuo epochų (GN={method}, greitis={speed})")
    plt.xlabel("Epocha")
    plt.ylabel("Paklaida")
    plt.legend()
    plt.grid(True)
    filename = f"paklaidos_{method}_sp{speed}_ep{ep}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_tikslumai(tikslumai_l, tikslumai_v, method, speed, ep):
    plt.plot(range(1, len(tikslumai_l)+1), tikslumai_l, label="Mokymas")
    plt.plot(range(1, len(tikslumai_v)+1), tikslumai_v, label="Validavimas")
    plt.title(f"Klasifikavimo tikslumo priklausomybė nuo epochų (GN={method}, greitis={speed})")
    plt.xlabel("Epocha")
    plt.ylabel("Tikslumas")
    plt.legend()
    plt.grid(True)
    filename = f"tikslumai_{method}_sp{speed}_ep{ep}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def stulp_greitis(paklaida, tikslumas, speed, method):
    x = np.arange(len(speed))
    width = 0.35

    plt.bar(x - width / 2, paklaida, width, label='Paklaida')
    plt.bar(x + width / 2, tikslumas, width, label='Tikslumas')

    plt.xlabel('Mokymo greitis')
    plt.ylabel('Reikšmė')
    plt.title('Testavimo duomenų galutinė paklaida ir klasifikavimo tikslumas')
    plt.xticks(x, speed)  # x ašies labeliai
    plt.legend()
    plt.grid(axis='y')
    filename = f"bar_{method}_sp{speed}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def stulp_lyginti(paklaidos, tikslumai):
    methods = ['Paketinis', 'Stochastinis']
    x = np.arange(len(methods))
    width = 0.35

    plt.bar(x - width / 2, paklaidos, width, label='Paklaida')
    plt.bar(x + width / 2, tikslumai, width, label='Tikslumas')

    plt.ylabel('Reikšmė')
    plt.title('Testavimo duomenų galutinė paklaida ir klasifikavimo tikslumas')
    plt.xticks(x, methods)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(axis='y')
    filename = "bar_lyginti.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


# --- EKSPERIMENTAI ---
def paleisti_eksperimentus(X_learn, y_learn, X_val, y_val, X_test, y_test, epoch_list, speed_list):
    rezultatai = []

    for method in ["stochastinis", "paketinis"]:
        for ep in epoch_list:
            for sp in speed_list:
                if method == "stochastinis":
                    start = time.time()
                    rez = stochastinis(X_learn, y_learn, ep, sp, X_val, y_val)
                    end = time.time()
                else:
                    start = time.time()
                    rez = paketinis(X_learn, y_learn, ep, sp, X_val, y_val)
                    end = time.time()

                w = rez["w"]

                galutine_test_err = galutine_paklaida(X_test, y_test, w)
                galutinis_test_acc = tikslumas(X_test, y_test, w)

                rezultatai.append({
                    "Metodas": method,
                    "Epochos": ep,
                    "Greitis": sp,
                    "Mokymo paklaida (test data)": galutine_test_err,
                    "Mokymo tikslumas (test data)": galutinis_test_acc,
                    "Laikas": end - start
                })

                paklaidos_l = rez['mokymo_paklaidos']
                paklaidos_v = rez['validavimo_paklaidos']
                tikslumai_l = rez['mokymo_tikslumai']
                tikslumai_v = rez['validavimo_tikslumai']

                plot_paklaidos(paklaidos_l, paklaidos_v, method, sp, ep)
                plot_tikslumai(tikslumai_l, tikslumai_v, method, sp, ep)

    return pd.DataFrame(rezultatai)


ats = paleisti_eksperimentus(X_learn, y_learn, X_validate, y_validate, X_test, y_test, [100, 1000, 5000], [0.0001, 0.005, 0.02])

print(ats.to_string())


# --- LYGINTI PAGAL GREIČIUS ---
# PAKETINIS
speed = [0.0001, 0.005, 0.02]
a1 = paketinis(X_learn, y_learn, 5000, 0.0001, X_validate, y_validate)
a2 = paketinis(X_learn, y_learn, 5000, 0.005, X_validate, y_validate)
a3 = paketinis(X_learn, y_learn, 5000, 0.02, X_validate, y_validate)

paklaida = [galutine_paklaida(X_test, y_test, a1['w']), galutine_paklaida(X_test, y_test, a2['w']), galutine_paklaida(X_test, y_test, a3['w'])]
tikslumas_1 = [tikslumas(X_test, y_test, a1['w']), tikslumas(X_test, y_test, a2['w']), tikslumas(X_test, y_test, a3['w'])]

stulp_greitis(paklaida, tikslumas_1, speed, 'Paketinis')


# STOCHASTINIS
b1 = stochastinis(X_learn, y_learn, 5000, 0.0001, X_validate, y_validate)
b2 = stochastinis(X_learn, y_learn, 5000, 0.005, X_validate, y_validate)
b3 = stochastinis(X_learn, y_learn, 5000, 0.02, X_validate, y_validate)

paklaida = [galutine_paklaida(X_test, y_test, b1['w']), galutine_paklaida(X_test, y_test, b2['w']), galutine_paklaida(X_test, y_test, b3['w'])]
##tikslumas_2 = [tikslumas(X_test, y_test, b1['w']), tikslumas(X_test, y_test, b2['w']), tikslumas(X_test, y_test, b3['w'])]

stulp_greitis(paklaida, tikslumas_2, speed, 'Stochastinis')


# --- GERIAUSIAS ATVEJIS ---
rez_p = paketinis(X_learn, y_learn, 5000, 0.02, X_validate, y_validate)

rez_s = stochastinis(X_learn, y_learn, 5000, 0.0001, X_validate, y_validate)

tikslumas_p = tikslumas(X_test, y_test, rez_p['w'])
paklaida_p = galutine_paklaida(X_test, y_test, rez_p['w'])

tikslumas_s = tikslumas(X_test, y_test, rez_s['w'])
paklaida_s = galutine_paklaida(X_test, y_test, rez_s['w'])

paklaidos = [paklaida_p, paklaida_s]
tikslumai = [tikslumas_p, tikslumas_s]
stulp_lyginti(paklaidos, tikslumai)

# --- PO MOKYMO ---
print('PAKETINIS')
print("Paskutinė mokymo paklaida:", rez_p["mokymo_paklaidos"][-1])
print("Paskutinė validavimo paklaida:", rez_p["validavimo_paklaidos"][-1])
print("Paskutinis mokymo tikslumas:", rez_p["mokymo_tikslumai"][-1])
print("Paskutinis validavimo tikslumas:", rez_p["validavimo_tikslumai"][-1])

print('STOCHASTINIS')
print("Paskutinė mokymo paklaida:", rez_s["mokymo_paklaidos"][-1])
print("Paskutinė validavimo paklaida:", rez_s["validavimo_paklaidos"][-1])
print("Paskutinis mokymo tikslumas:", rez_s["mokymo_tikslumai"][-1])
print("Paskutinis validavimo tikslumas:", rez_s["validavimo_tikslumai"][-1])

# Klasių palyginimas
def gautos_klases(X, w):
    gauta = []
    for i in range(len(X)):
        a = sum(w[j] * X[i, j] for j in range(X.shape[1]))
        f = round(sigmoidine(a))
        gauta.append(f)
    return gauta


gauta_p = gautos_klases(X_test, rez_p['w'])
gauta_s = gautos_klases(X_test, rez_s['w'])
print(f'Paketinis 0.02, 5000ep:\nGauta: {gauta_p}\norg: {y_test}')
print(f'Stochastinis 0.0001, 5000ep:\nGauta: {gauta_s}\norg:{y_test}')


