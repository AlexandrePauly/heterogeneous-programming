# Heterogeneous Programming

## Architecture
- data/ : Les images utilisées.
- doc/ : Les consignes du TP.
- src/ : Les fichiers sources d'implémentation de la solution (en C et Cuda).

## Instructions d'installation

1. Clonez ce dépôt sur une machine virtuelle.

2. Exécutez les commandes suivantes :

```bash
cd heterogeneous-programming
```

Pour tester le code en c :

```bash
cd src/C
gcc -o exe exemple.c -lm
./exe ../../data/lena_original.png output/final_lena.png
```
Pour tester le code en cuda :

```bash
cd src/Cuda
nvcc -o exe exemple.c lodepng.cpp
./exe ../../data/lena_original.png output/final_lena.png
```

>**_Attention :_** Évitez de modifier l'arborescence du projet pour ne pas casser les url !

## Installer from scratch si besoin pour Cuda 

```bash
vwget https://raw.githubusercontent.com/lvandeve/lodepng/master/lodepng.h

wget https://raw.githubusercontent.com/lvandeve/lodepng/master/lodepng.cpp

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb

dpkg -i cuda-keyring_1.0-1_all.deb

apt upgrade

cd src/Cuda

nvcc -o exe exemple.c lodepng.cpp

./exe ../../data/lena_original.png output/final_lena.png
```

## Auteur

Réalisé par Alexandre PAULY et Laura SABADIE.

## Licence

Vous êtes libre de l'utiliser, le modifier et le distribuer.

## Contributions

Les contributions à ce projet sont les bienvenues. N'hésitez pas à ouvrir une demande d'extraction pour proposer des améliorations, des corrections de bugs ou de nouvelles fonctionnalités.

## Remarques

Veuillez noter que cette application est destinée à des fins éducatives et expérimentales ayant permis de renforcer des connaissances dans différents langages de programmation.
