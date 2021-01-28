import sys
import pandas as pd
import numpy as np
import face_recognition as fr
import cv2
import os
import pickle
import keyboard
import pyautogui
import time
import platform
import matplotlib.pyplot as plt
from distlib.compat import raw_input
from math import floor

# initialise les repertoire des fichiers
global PATH_IMAGE
global PATH_DATA
global PATH_TEST
global known_face_encodings
global known_face_names

PATH_IMAGE = "known_emotions/"
PATH_DATA = "data/"
PATH_TEST = "test/"

# initialise les tablreau liant les images au émotion
known_face_encodings = []
known_face_names = []


# fonction de fin de programme
def retourMenu():
    print("Appuyer sur <space> pour revenir au menu principale\n")
    while not keyboard.is_pressed('space'):
        time.sleep(0.0001)


def diagramme(idSentiment, data):
    plt.bar(idSentiment, data,
            color=['#5cb85c', '#5bc0de', '#d9534f', '#1C649A', '#429A1C', '#5D1C9A', '#9A4C1C', '#DF7156'], width=0.8)
    plt.xlabel("Emotion(s)")
    plt.ylabel("Pourcentage (%)")
    plt.title("Pourcentages de reussites du test :")
    plt.show()


def testRecoVisageCamera():
    # Charge les données precedement encodé
    with open(PATH_DATA + 'emotion.data', 'rb') as f:
        all_face_encodings = pickle.load(f)
    known_face_encodings = all_face_encodings["encodings"]  # encoding image column
    known_face_names = all_face_encodings["names"]  # emotion names column
    print('[INFO] Faces well imported')

    # webcam processing thread
    print('[INFO] Starting Webcam...')
    # demare la capture video
    video_capture = cv2.VideoCapture(0)

    print("[INFO] Webcam Started.\nAppuyer sur 'q' pour quitter.")
    while True:
        ret, frame = video_capture.read()
        # Revoie l'image au format RGB (et non BGR)
        rgb_frame = frame[:, :, ::-1]
        # Renvoie un tableau d'image englobantes un visages humains
        face_locations = fr.face_locations(rgb_frame)
        # Étant donné une image, renvoie le codage de visage de 128 dimensions pour chaque visage de l'image.
        face_encodings = fr.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Comparez une liste d'encodages de visage à un encodage candidat pour voir s'ils correspondent.
            matches = fr.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            # À partir de la liste d'encodages de visage, le compare à un encodage de visage connu et obtenez une
            # distance euclidienne pour chaque face de comparaison. La distance indique à quel point les visages
            # sont similaires.
            face_distances = fr.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                # associe avec le nom de l'émotion la plus proche
                name = known_face_names[best_match_index]
            # créé le rectangle autour du visage
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            # creé un sous text avec le nom de l'emotion à l'interieur
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            print('[DETECTION] ' + name)
        # affiche l'image
        cv2.imshow('Webcam - Facial Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('[INFO] Stopping System')
            break

    video_capture.release()
    cv2.destroyAllWindows()
    pyautogui.press('enter')


def testRecoEmotion():
    # Charge les données precedement encodé
    global total_img, success
    success = 0
    print('[INFO] Importing data...')
    with open(PATH_DATA + 'emotion.data', 'rb') as f:
        all_face_encodings = pickle.load(f)
    known_face_encodings = all_face_encodings["encodings"]  # encoding image column
    known_face_names = all_face_encodings["names"]  # emotion names column
    print('[INFO] Data well imported')

    # on commence par choisir le test à effectuer
    action2 = ""
    while action2 != "9":
        print("Choissiser le test à effectuer :\n")
        action2 = raw_input("\t1) Tester toutes les émotions.\n\t2) Tester pour une emotion.\n\t9) Retour.\n")

        if action2 == '1' or action2 == '2':
            action3 = raw_input("Choisiser le type de test :\n\t1) Test cour (100 images/émotions)."
                                "\n\t2) Test long (1000 images/émotions, quand possible)\n\t9) Quitter\n")
            if action3 == "9":
                break

            # Si action2 est égale à "1", on test toutes les emotions
            if action2 == "1":

                # on initialise un tableau de 2x2 avec un emotion lier on nombre d'image traité pour celle-ci
                nb_img = [["Colere", 0], ["Degout", 0], ["Heureux", 0], ["Neutre", 0], ["Peur", 0], ["Surpris", 0],
                          ["Triste", 0]]

                # on initialise pour chaque emotion un compteur qui va s'incrementer à chaque fois que le programme
                trueColere = 0
                trueDegout = 0
                trueHeureux = 0
                trueNeutre = 0
                truePeur = 0
                trueSurpris = 0
                trueTriste = 0

                test_dir = os.listdir(PATH_TEST)

                # Loop through each emotion in the training directory
                index = -1

                print("[INFO] Debut du test")
                # regarde les 7 emotions du dossier "test_dir"
                for emotion in test_dir:

                    # emotion contiendra par exemple "Colere/"
                    emotion += "/"
                    index += 1
                    total_img = 0
                    print("[INFO] Emotion '", nb_img[index][0], "' en cours...")
                    time.sleep(1)
                    pix = os.listdir(PATH_TEST + emotion)

                    # Parcour chaque image de test pour l'émotion actuelle
                    for emotion_img in pix:
                        nb_img[index][1] += 1
                        total_img += 1

                        # Obtient les encodages de visage pour le visage dans chaque fichier image
                        frame = fr.load_image_file(PATH_TEST + emotion + emotion_img)

                        # met dans rgb_frame l'image frame en couleur (si se n'est pas deja le cas)
                        rgb_frame = frame[:, :, ::-1]

                        face_locations = fr.face_locations(rgb_frame)
                        face_encodings = fr.face_encodings(rgb_frame, face_locations)
                        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                            # Comparez une liste d'encodages de visage à un encodage candidat pour voir s'ils correspondent.
                            matches = fr.compare_faces(known_face_encodings, face_encoding)
                            name = "Unknown"
                            # À partir d'une liste d'encodages de visage, les compare à un encodage de visage connu et obtenez une
                            # distance euclidienne pour chaque face de comparaison. La distance indique à quel point les visages
                            # sont similaires.
                            face_distances = fr.face_distance(known_face_encodings, face_encoding)
                            best_match_index = np.argmin(face_distances)

                            if matches[best_match_index]:
                                # associe avec le nom de l'émotion la plus proche
                                name = known_face_names[best_match_index]
                            # créé le rectangle autour du visage
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                            # creé un sous text avec le nom de l'emotion à l'interieur
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                            print("Image :", nb_img, '\n[DETECTION] ', name)
                            # attribue le resultat du test unitaire en incrementant si l'emotion à été reconnue correctement
                            # 0=colere 1=degout 2=heureux 3=neutre 4=peur 5=surpris 6=triste
                            if index == 0 and name == "Colere":
                                trueColere += 1
                            if index == 1 and name == "Degout":
                                trueDegout += 1
                            if index == 2 and name == "Heureux":
                                trueHeureux += 1
                            if index == 3 and name == "Neutre":
                                trueNeutre += 1
                            if index == 4 and name == "Peur":
                                truePeur += 1
                            if index == 5 and name == "Surpris":
                                trueSurpris += 1
                            if index == 6 and name == "Triste":
                                trueTriste += 1

                        # condition s'il on veut effectuer un "test court",
                        # et que l'emotion ectuelle à traiter 100 image, arrete le test
                        if total_img == 100 and action3 == "1":
                            break
                    print("[INFO] Emotion '", nb_img[index][0], "' terminé (", index + 1, "/ 7 ).")

                print("[INFO] fin du test")

                # les stats commence ici
                Succes_totaux = trueDegout + trueHeureux + trueNeutre + truePeur + trueTriste + trueSurpris + trueColere
                perCentSuccesTo = round((Succes_totaux * 100 / total_img) / 7)
                perCentSuccesCo = round(trueColere * 100 / nb_img[0][1])
                perCentSuccesDe = round(trueDegout * 100 / nb_img[1][1])
                perCentSuccesHe = round(trueHeureux * 100 / nb_img[2][1])
                perCentSuccesNe = round(trueNeutre * 100 / nb_img[3][1])
                perCentSuccesPe = round(trueNeutre * 100 / nb_img[4][1])
                perCentSuccesSu = round(trueSurpris * 100 / nb_img[5][1])
                perCentSuccesTr = round(trueTriste * 100 / nb_img[6][1])
                idSentiment = ["Colère", "Degout", "Heureux", "Neutre", "Peur", "Surpris", "Triste", "Succes Total"]
                data = [perCentSuccesCo, perCentSuccesDe, perCentSuccesHe, perCentSuccesNe,
                        perCentSuccesPe, perCentSuccesSu, perCentSuccesTr, perCentSuccesTo]

                diagramme(idSentiment, data)

                # On affiche les resultats du test
                print(pd.Series(data, idSentiment))

                # On redirige l'entré standard vers le fichier "resultat.txt" pour y ecrir les résultats
                # On enregistre la sortie standard courante
                sortie = sys.stdout
                # on redirige
                sys.stdout = open('resultat.txt', 'a')
                # on ecrit
                print("Resultat avec", total_img, " images de tests sur toutes les émotions.\n")
                print(pd.Series(data, idSentiment))
                print("\n")
                # on redirige vers l'ancienne sortie
                sys.stdout = sortie

                retourMenu()

            # Si action2 est égale à "2", on test une émotion
            if action2 == '2':
                pix = os.listdir(PATH_TEST)
                emotion = "na"
                nb_img = 0

                success = 0
                t = False
                while emotion not in pix:
                    print("Taper l'émotion dont vous voulez effectuer :\n")

                    # L'utilisateur choisie l'émotion à tester
                    emotion = raw_input(
                        " Colere | Degout | Heureux | Neutre | Peur | Surpris | Triste | ou vide pour annuler\n")
                    if emotion == "":
                        t = True
                        break
                    if emotion not in pix:
                        print('Erreur de syntaxe,\n')
                # permet de sortir des 2 while pour revenir au menu principal
                if t:
                    return

                # Traite l'émotion choisie
                emotion += "/"
                pix = os.listdir(PATH_TEST + emotion)
                print("[INFO] Debut du test")
                for emotion_img in pix:
                    nb_img += 1

                    # Obtient les encodages de visage pour le visage dans chaque fichier image
                    frame = fr.load_image_file(PATH_TEST + emotion + emotion_img)

                    # met dans rgb_frame l'image frame en couleur (si se n'est pas deja le cas)
                    rgb_frame = frame[:, :, ::-1]

                    face_locations = fr.face_locations(rgb_frame)
                    face_encodings = fr.face_encodings(rgb_frame, face_locations)

                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        # Comparez une liste d'encodages de visage à un encodage candidat pour voir s'ils correspondent.
                        matches = fr.compare_faces(known_face_encodings, face_encoding)
                        name = "Unknown"
                        # À partir d'une liste d'encodages de visage, les compare à un encodage de visage connu et obtenez une
                        # distance euclidienne pour chaque face de comparaison. La distance indique à quel point les visages
                        # sont similaires.
                        face_distances = fr.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)

                        if matches[best_match_index]:
                            # associe avec le nom de l'émotion la plus proche
                            name = known_face_names[best_match_index]
                        # créé le rectangle autour du visage
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                        # creé un sous text avec le nom de l'emotion à l'interieur
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                        print("Image :", nb_img, '\n[DETECTION] ', name)

                        # Si l'émotion reconnue est égale à l'émotion demandé, incremente de 1
                        if name + '/' == emotion:
                            success += 1
                    if nb_img == 100 and action3 == "1":
                        break
                print("[INFO] fin du test")

            # affiche les stats
            perCentSucces = round(success * 100 / nb_img)
            print("Nombre d'images :", nb_img, "Pourcentage de succes :", success)
            idSentiment = [emotion]
            data = [perCentSucces]

            diagramme(idSentiment, data)

            print(pd.Series(data, idSentiment))

            # On redirige l'entré standard vers le fichier "resultat.txt" pour y ecrir les résultats
            # On enregistre la sortie standard courante
            sortie = sys.stdout

            # On redirige la sortie standard vers le fichier 'resultat.txt'
            sys.stdout = open('resultat.txt', 'a')

            # On écrit
            print("Resultat avec", nb_img, " images de tests sur l'émotion ", emotion, ".\n")
            print(pd.Series(data, idSentiment))
            print("\n")
            # On retatblie la sortie standard
            sys.stdout = sortie
            
            retourMenu()

        print("Fermeture...")
        time.sleep(1)
        break


def entrainement():
    print("\t\t\t/!\ attention /!\ \nCette operation peut prendre beaucoup de temps (plusieurs heures) et ne doit "
          "pas etre interrompue, êtes-vous sur de vouloir continuer ?(y/n)")
    if raw_input() == 'y':
        if raw_input("Dernière avertissement !!\n"
                     "Si vous voulez continuer, taper 'oui, on lance'\n") == 'oui, on lance':
            timeStart = time.perf_counter()
            print('[INFO] Starting training...')
            train_dir = os.listdir(PATH_IMAGE)

            # Parcoure chaque émotion dans le répertoire de formation
            for emotion in train_dir:
                pix = os.listdir(PATH_IMAGE + emotion)
                # Parcoure chaque image d'entrainement pour l'émotion actuelle
                for emotion_img in pix:
                    # Obtien l'encodages du visage pour chaque fichier image
                    face = fr.load_image_file(PATH_IMAGE + emotion + "/" + emotion_img)
                    # retourne les visages encodés
                    face_bounding_boxes = fr.face_locations(face)

                    # Si l'image d'entraînement contient exactement un visage
                    if len(face_bounding_boxes) == 1:
                        face_enc = fr.face_encodings(face)[0]

                        # encodage le visage pour l'image actuelle
                        # avec l'étiquette (name) correspondant aux données d'entraînement
                        known_face_encodings.append(face_enc)
                        known_face_names.append(emotion)
                        print(emotion + "/" + emotion_img + " is encoding")
                    # Sinon supprime l'image du dossier pour eviter d'avoir à la re-traiter lors d'un prochain entrainement
                    else:
                        os.remove(PATH_IMAGE + emotion + "/" + emotion_img)
                        print(emotion + "/" + emotion_img + " is delete")

            # creer un tableau contenant les 2 listes : la premiere contient les encodages des images et
            # la deuxieme, les correspondance avec l'emotion
            data = {"encodings": known_face_encodings, "names": known_face_names}
            # ouvre le fichier "emotion.data"
            f = open("emotion.data", "wb")
            # insert dans le fichier toute les données
            f.write(pickle.dumps(data))
            # ferme le fichier
            f.close()
            print('[INFO] Training completed...')
            print("Temps total d'execution :\n")

            # affichage d'un timer
            timeT = time.perf_counter() - timeStart
            hours = floor(timeT / 3600)
            timeT -= 3600 * hours
            min = floor(timeT / 60)
            timeT -= 60 * min
            print(hours, "heures :", min, "minutes : ", floor(timeT), "secondes\n")
            retourMenu()
        else:
            print("[INFO] Fermeture...")
            time.sleep(1)
    else:
        print("[INFO] Fermeture...")
        time.sleep(1)


# Le programme se decompose en 4 partie :
# (1) Test en direct de la reconnaissance d’émotion avec la webcam
#
# (2) Test avec des images d'entraînement et affiche les statistiques par émotion
#
# (3) Entraîne la base de données avec de nouvelles images, les classe par émotion et serialize la base de donnée au format binaire.
#
# (4)  Ferme le programme
#
# Chaque partie est expliqué plus bas, sauf (4) qui ferme le programme et ne requière pas plus d'explication
#
# initialise la premiere action de l'utilisateur à vide
action = ""
while action != "9":

    # attend que l'utilisateur entre un nombre
    action = input("Que voulez-vous faire : \n\t 1) Des tests avec votre webcam.\n\t 2) Des tests avec les images "
                   "de test.\n\t 8) Lancer un entrainement.\n\t 9) Quitter.\n")

    # si l'action est égale à "1", lance la reconnaisance à l'aide de la webcam de l'ordinateur
    # Cela donne un coté ludique au projet.
    # Attention cependant, la reconnaissance d'emotion se base sur les traits du visage à partir de photo.
    # il est donc très probable que le programme "interprète" vos émotions de façon differente de la votre
    # et cette experience, bien qu'amusante, ne constitue pas un test fiable de la capacité ou non du programme
    # à reconnaitre une émotion :
    # Pour cela passer à l'action "2"

    if action == "1":
        testRecoVisageCamera()

    # Si l'action est égale à "2", lance les tests de reconnaissance d'émotion:
    # Plusieurs possibilité s'offre à l'utilisateur :
    #       -tester une émotion
    #       -tester toutes les émotions
    #
    # Ensuite, il aura encors le choix de faire un test long ou cours :
    #
    #       Les tests court sont moins fiable car il utilise que 100 image par émotion
    #       mais dure 1-2 min pour une émotion et 15-20 min pour toute les emotionss.
    #       C'est en quelque sorte un test de "démonstration"
    #
    #       les tests durent plus longtemps, 15-20 min pour une émotion et 1H en moyenne pour faire un test complet,
    #       mais il est beaucoup plus fiable que le test court du fait que l'on test 1 000 images par émotions
    #
    # À la fin du programme, les resultat des tests seront enregistré dans une fichier "resultat.txt"

    if action == "2":
        testRecoEmotion()

    # Si l'action est égale à 8
    # On lance un entrainement
    if action == '8':
        entrainement()
    #
    # permet de 'clear' la console
    if platform.system() == "Windows":
        os.system('cls')
        pyautogui.hotkey('ctrl', 'l')
    else:
        os.system('clear')
        pyautogui.hotkey('ctrl', 'l')
    pyautogui.hotkey('ctrl', 'l')
