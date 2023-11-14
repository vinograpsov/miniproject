# Miniprojekt - hypernetwork

Celem dzisiejszych zajec ejst wprowadzenie pojecia hypernetworkow. 
Bedzie zaprezentowane podsawowe podejscie do hypernetworkow,
oraz zaimplementowana prosta siec, ktora bedzie wykorzystywac hypernetworki dla latwijszego porazumienia ich dzialania.

PLAN LEKCJI

1. INTRODUCTION 
   - Wprowadzenie do hypernetworkow 
   - Przyklad prostej integracji hypernetworkow do sieci neuronowej
2. Categorization of Hypernetworks
   - teoretyczne podejscie do roznych typow hypernetworkow
   - ....
3. Mozliwosci modelu 
    - When can we use Hypernets?
    ..... 
4. Hypernet + Resnet 
5. Hypernet + MLP ???? 
6. Podsumowanie
    - Miejsce hypernetworkow w swiecie sieci neuronowych
    - Wady i zalety
    - zadanie domowe

Repozytorium zawiara pliki: 
- hypernet.ipynb -- wprowadzenie do tematyki hypernetworkow na przykladzeie zbirow treningowych MNIST, oraz raspatrzenie  
- requirements.txt -- plik z wymaganymi bibliotekami do uruchomienia notebooka
- imgs -- folder zawierajacy obrazki wykorzystywane w notebooku

## Konfiguracja srodowiska

Aby uruchomic notebooka nalezy zrobic environment oraz zainstalować niezbędne biblioteki z pliku `requirements.txt
Zaleca się przygotowanie wcześniej wirtualnego środowiska:

Stwórz środowisko za pomocą `venv`:
```bash
$ python3.10 -m venv .venv
```
lub z użyciem `conda`:
```bash
$ conda create -n .venv python=3.10
```


zainstaluj niezbędne biblioteki:
```bash
$ source .venv/bin/activate
$ pip install -r requirements.txt
```


## Źródła:

Repozytorium zostało stworzone na podstawie:<br/> 
[1] https://github.com/g1910/HyperNetworks [12.10.2023]<br/>
[2] https://github.com/chrhenning/hypnettorch [12.10.2023]<br/>
[3] A BRIEF REVIEW OF HYPERNETWORKS IN DEEP LEARNING, Vinod Kumar Chauhan, Jiandong Zhou, Ping Lu, Soheila Molaei, David A. Clifton [11.07.2023]