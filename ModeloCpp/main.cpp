#include "EXTRACCION/extracciondata.h"
#include "REGRESIONLINEAL/regresion_lineal.h"
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <boost/algorithm/string.hpp>



int main(int argc, char* argv[]){
    /* Se crea el objeto de tipo "extracciondata"
     * y se incluyen los 3 argumentos que hemos de
     * pasar al objeto (dado por el constructor de
     * la clase) */

    /*se instancia el objeto de la clase extraccion a extraer*/
    ExtraccionData extraer(argv[1], argv[2],argv[3]);

    /*se instancia el objeto de la clase RegresionLineal a RL*/
    RegresionLineal RL;

    /* Se leen los datos del fichero, a traves
     * de la  función ReadCSV() */

    std::vector<std::vector<std::string>> dataSET = extraer.ReadCSV();



    int filas = dataSET.size() + 1;
    int columnas = dataSET[0].size();

    Eigen::MatrixXd DataFrame = extraer.CSVtoEigen(dataSET, filas, columnas);

    Eigen::MatrixXd mean = extraer.Promedio(DataFrame);

    std::cout << DataFrame << std::endl;
    std::cout <<"\n-------------------------------Promedio----------------------------------------\n"<< std::endl;
    std::cout <<mean<< std::endl;


    /*El objeto CVSto Eigen (simila a un objeto Dataframe)
    * se normaliza: se optiene una matriz matNormal */
    Eigen::MatrixXd matNormal = extraer.Normalizador(DataFrame);
    /*std::cout<<matNormal<<std::endl<<std::endl;*/

    /* A continuación se hará el primer modulo de Machine Learning:
     * se requiere una clase de Regresión Lineal (Implementacion e Interfaz),
     * debe definir un constructor, importar las bibliotecas necesarias. Se debe tener en cuenta
     * que el método de Regresión lineal es un método estadístico que define la relacion entre las
     * variables independientes, con la variable dependiente. la idea
     * principal, es definir una LINEA RECTA  (Hiperplano) con sus
     * correspondientes coeficientes (pendientes) y los puntos de corte (y=0).
     * Se tiene diferentes metodos para resolver RL: se implementara
     * el metodo de los Minimos Cuadrados Ordinarios (OLS). El OLS
     * es un método sencillo y computracionalmente económico. Ols
     * presenra una solución óptima para conjuntos de datos complejos
     * Para el PRESENTE caso, se tiene un conjunto de datos ()
     * con multiples variables independientes. Se necesita el algoritmo
     * llamado GRADIENTE DESCENDIENTE.El objetivo del GD es minimizar la
     * "Función de costo" */

     Eigen::MatrixXd X_train, y_train, X_test, y_test;

     /* Declaramos un objeto para recibir la tupla empaquetada */
     std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> empaquetado = extraer.TrainTestDiv(matNormal,0.8);

     /* se necesita desempaquetar la tupla en 4 grupos de datos */
     std::tie(X_train, y_train, X_test, y_test) = empaquetado;

     /*se imprime total de filas, las filas de entrenamiento,
      * las filas para pruebas en sus dos sabores*/
   /**/std::cout<<matNormal.rows()<<std::endl;
     std::cout<<X_train.rows()<<std::endl;
     std::cout<<y_train.rows()<<std::endl;
     std::cout<<X_test.rows()<<std::endl;
     std::cout<<y_test.rows()<<std::endl;


     /* Se necesita imprimir las cantidad de columnas por sabor */

   /*std::cout<<matNormal.cols()<<std::endl;
     std::cout<<X_train.cols()<<std::endl;
     std::cout<<y_train.cols()<<std::endl;
     std::cout<<X_test.cols()<<std::endl;
     std::cout<<y_test.cols()<<std::endl;
   */


    /* Se tiene en cuenta que la regresion lineal es un metodo
     * estadistico, la idea principal es crear un hiper plano
     * con tantas dimesiones como variable independientes tenga
     * el dataset(Pendientes-pesos(m), puntos de corte(b)).
     * se hace la prueba del modelo: Se crea un vector para prueba
     * y entrenamiento inicializado en "unos", que corresponde
     * a nuestras variables independientes (futurs)*/
     Eigen::VectorXd vectorTrain = Eigen::VectorXd::Ones(X_train.rows());
     Eigen::VectorXd vectorTest = Eigen::VectorXd::Ones(X_test.rows());

    /* Se redimensiona las matrices para ser ubicadas en los vectores
     * anteriores, Similar a reshape de numpy */
     X_train.conservativeResize(X_train.rows(), X_train.cols()+1);
     X_train.col(X_train.cols()-1)=vectorTrain;

     X_test.conservativeResize(X_test.rows(), X_test.cols()+1);
     X_test.col(X_test.cols()-1)=vectorTest;

    /* Se define eñ vector theta, para pasar el alortimo del GD,
     * basicamento es un vector de 0 del mismo tamaño de entrenamiento,
     * adicional se declara alfa y el numero de iteraciones*/
     Eigen::VectorXd vectorTheta = Eigen::VectorXd::Zero(X_train.cols());
     float alfa = 0.01;
     int iterator = 1000;

     Eigen::VectorXd thetaOut;
     std::vector<float>costo;


     std::tuple<Eigen::VectorXd, std::vector<float>> salidaGD = RL.Gradiente(X_train, y_train,vectorTheta,alfa,iterator);
     std::tie(thetaOut, costo) = salidaGD;
     /* std::cout<<thetaOut<<std::endl; */

    /* Se quiere observar como decrese la función de costo */
    /*for(auto v: costo){
         std::cout<<v<<std::endl;
     }*/

     /* Acontinuación por propositos de seguridad se exportan el vector de costo y el
      * vector theta a ficheros*/
     extraer.vectorToFile(costo, "vectorCosto.txt");
     extraer.matrixToFile(thetaOut, "vectorTheta.txt");

    /*con el proposito de ajustar el modei8 c*/
     auto MuPromedio= extraer.Promedio(DataFrame);
     auto MuFeatures= MuPromedio(0,6);
     auto EscaladaData= DataFrame.rowwise()-DataFrame.colwise().mean();
     auto MuEstandar= extraer.DesvStandard(EscaladaData);
     auto DevFeatures= MuEstandar(0,6);
     Eigen::MatrixXd y_train_hat= (X_train*thetaOut*DevFeatures).array()+MuFeatures;
     Eigen::MatrixXd y= DataFrame.col(6).topRows(1070);

     /*A continuacion se determina que tan bueno es nuestro modelo
      * utilizando la metrica R2*/

      float ComprobacionMetrica= RL.R2(y, y_train_hat);
      std::cout<<ComprobacionMetrica<<std::endl;

      extraer.matrixToFile(y_train_hat, "Prediccion.txt");












    return EXIT_SUCCESS;
}

