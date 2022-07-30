import numpy as np
import cv2 as cv
import glob

#tamanho do tabuleiro
chessBoardSize = (9, 6)

# critério de término padrão do OpenCV
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#prepara os object ponints (pontos em 3D) 
objp = np.zeros((chessBoardSize[1]*chessBoardSize[0],3), np.float32)
objp[:,:2] = np.mgrid[0:chessBoardSize[0],0:chessBoardSize[1]].T.reshape(-1,2)

#arrays para guardar os pontos 3D das imagens
objpoints = [] #pontos 3D no mundo real
imgpoints = [] #pontos 2D no plano da imagem

print('Realizando leitura das imagens e detecção de pontos e cantos...')
#pega as imagens do diretório
images = glob.glob('*.jpg')
for fname in images:
    print(fname)
    if '-corners' in fname: #ignora as imagens cujos pontos já foram detectados
        continue
    if '-calib' in fname: #ignora as imagens cuja distorção já foi retirada
        continue

    #le a imagem
    img = cv.imread(fname)

    #transforma a imagem para padrões de cinza
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # procura os cantos do tabuleiro
    ret, corners = cv.findChessboardCorners(gray, (chessBoardSize[0],chessBoardSize[1]), None)

    # se encontrar os pontos 3D e 2D -> adiciona nos arrays
    if ret == True:
        print('Encontrou pontos na imagem!')
        objpoints.append(objp)

        #refina de acordo com o critério
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # desenha e mostra os cantos e pontos
        cv.drawChessboardCorners(img, (chessBoardSize[0],chessBoardSize[1]), corners2, ret)
        
        #salva a imagem
        cv.imwrite(fname.replace('.jpg', '') + '-corners.jpg', img)
        cv.waitKey(500)
    else:
        print('Não encontrou pontos na imagem!')

cv.destroyAllWindows()

print('Calibrando câmera...')
#agora calibra a camera de acordo com os pontos encontrados
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print('Retirando distorções das imagens...')
#retira as distorções das imagens
images = glob.glob('*.jpg')
for fname in images:
    print(fname)
    if '-corners' in fname: #ignora as imagens cujos pontos já foram detectados
        continue
    if '-calib' in fname: #ignora as imagens cuja distorção já foi retirada
        continue

    img = cv.imread(fname)
    h,  w = img.shape[:2]

    #refina a matriz da câmera
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    #tira a distorção
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    
    #salva imagem
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite(fname.replace('.jpg', '') + '-calib.jpg', dst)

