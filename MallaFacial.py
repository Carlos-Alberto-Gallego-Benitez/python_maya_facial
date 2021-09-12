import cv2
import mediapipe as mp
import math

cap = cv2.VideoCapture(0) # capturamos la camara del pc
cap. set(3, 6000)# ancho de la ventana
cap.set(3, 8000) #alto de la ventana

#-----------------------------------------------función para crear el dibujo------------------------------------------


mpDibujo = mp.solutions.drawing_utils
ConfiDibujo = mpDibujo.DrawingSpec(thickness=1, circle_radius=1)

#------------------------------------------para pintar el dibujo en el rostro----------------------------------------
mpMallaFacial = mp.solutions.face_mesh #instancia de la función que genera la malla facial
MallaFAcial = mpMallaFacial.FaceMesh(max_num_faces=1)



#------------------------------------------------------while principal------------------------------------------------
while True:

    ret, frame = cap.read()

    #------------------------correción de color-------------------
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #---------------------resultados--------------------------
    result = MallaFAcial.process(frameRGB)

    #---------creamos los arreglos que van a almacenar los resultados------------
    px = []
    py = []
    lista = []
    r = []
    t = []


    if result.multi_face_landmarks:
        for rostros in result.multi_face_landmarks:
            mpDibujo.draw_landmarks(frame, rostros, mpMallaFacial.FACE_CONNECTIONS, ConfiDibujo, ConfiDibujo)

            #-----------se extraen los puntos del rostro capturado------------------
            for id, puntos in enumerate(rostros.landmark):
                al, an, c = frame.shape
                x,y = int(puntos.x*an), int(puntos.y*al)
                px.append(x)
                py.append(y)
                lista.append([id, x, y])
                if len(lista) == 468:
                    #ceja derecha
                    x1, y1 = lista[65][1:]
                    x2, y2 = lista[158][1:]
                    cx, cy = (x1 + x2) // 2, (y1 + y2 ) // 2
                    longitud1 = math.hypot(x2 - x1, y2 - y1)

                    #ceja izquierda
                    x3, y3 = lista[295][1:]
                    x4, y4 = lista[385][1:]
                    cx2, cy2 = (x3 + x4) // 2, (y3 + y4) // 2
                    longitud2 = math.hypot(x4 - x3, y4- y3)

                    #boca extremos
                    x5, y5 = lista[78][1:]
                    x6, y6 = lista[308][1:]
                    cx3, cy3 = (x5 + x6) // 2, (y5 + y6) // 2
                    longitud3 = math.hypot(x6 - x5, y6-y5)

                    #boca apertura
                    x7, y7 = lista[13][1:]
                    x8, y8 = lista[14][1:]
                    cx4, cy4 = (x7 + x8) // 2, (y7 + y8) // 2
                    longitud4 = math.hypot(x8 - x7, y8 -y7)

                    #validamos el estado del rostro
                    #enojado
                    if longitud1 < 19 and longitud2 < 19 and longitud3 > 80 and longitud3 < 95 and longitud4 < 5:
                        cv2.putText(frame, 'Persona Enojada', (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    #feliz
                    elif longitud1 > 20 and longitud1 < 30 and longitud2 > 20 and longitud2 < 30 and longitud3 > 109 and longitud4 >10 and longitud4 < 20:
                        cv2.putText(frame, 'Persona Feliz', (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

                    #asombrado
                    elif longitud1 > 35 and longitud2 > 35 and longitud3 > 80 and longitud3 < 90 and longitud4 > 20:
                        cv2.putText(frame,'Persona Asombrada',(200, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                    #triste
                    elif longitud1 > 20 and longitud1 < 35 and longitud2 > 20 and longitud2 < 35 and longitud3 > 80 and longitud3 < 95 and longitud4 < 5:
                        cv2.putText(frame,'Persona Triste', (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)


    cv2.imshow("Reconocimiento de estado emocional", frame)
    t = cv2.waitKey(1)

    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()