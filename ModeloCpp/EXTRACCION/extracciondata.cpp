#include "extracciondata.h"
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <boost/algorithm/string.hpp>


/*SE implementa la primera dunción miembro: lectura
* del fichero CSV. para ello disponemos de
* vector de vectores  del tipo string, en donde
* se itera linea por linea y se almacena el vector
* de vectores del tipo string, cada registro o fila
*/

std::vector<std::vector<std::string>> ExtraccionData::ReadCSV(){
    /*Se abre el fichero .csv */

    std::ifstream Fichero(setDatos);

    /**/
    std::vector<std::vector<std::string>> datosString;
    /* Se itera a traves de cada linea. Se divide el
     * contenido según el delimitador provisto por el
     * constructor */
    std::string linea = "";
    while(getline(Fichero,linea)) {
        /*cada linea se almacena en vectorFila*/
        std::vector<std::string> vectorFila;
        /*cada vector se divide segun delimitador*/
        boost::algorithm::split(vectorFila,linea,boost::is_any_of(delimitador));
        /*cada fila se ingresa al vector de vectores*/
        datosString.push_back(vectorFila);
    }
    /*Se cierra el fichero*/
    Fichero.close();

    return datosString;

}

/* Segunda funcion miembr: Almacenar el vector de vectores
del tipo string en una matrix, la idea es presentar el
conjunto de datos similar a un objeto pandas (Dataframe)*/

Eigen::MatrixXd ExtraccionData::CSVtoEigen(
        std::vector<std::vector<std::string>> setDatos,
        int filas, int columnas){
    /*Identificar si tiene o no cabecera*/
    if(header==true){
        filas = filas - 1;
    }
    /*Se itera sobre la filas y columnas, para almacenar/
     * em la matrix de dimensión filasxcolumnas.
     * Basicamente, se almacenará strings del vector:
     * que luego se pasa a "float" para ser manipulados */

    Eigen::MatrixXd dfMatriz(columnas, filas);
    int i,j;
    for(i=0;i<filas;i++)
        for(j=0;j<columnas;j++)
            dfMatriz(j,i) = atof(setDatos[i][j].c_str()); //Se al acena del tipo float

    /* Se transpone la matriz para retornar */
    return dfMatriz.transpose();
}

/* Se requiere implementar una funcion que calcule el
 * promedio de los datos (xcoluumnas). La funcion debe
 * ser verificada con python usando cualquier
 * biblioteca (pandas, sklearn, seaborn...).
 * En c++, existe el tipo de dato "auto" -> "decltype"
 * En muchos casos, la herencia del tipo de dato no es
 * evidente. El tipo de dato auto -> decltype
 * especifica el tipo de variable (reduce en tiempo de
 * compilación) que va a heredar la función. es decir,
 * en la función, si el tipo de retorno es "auto", se
 * evaluara mediante la expresión para la dedcción del
 * tipo de dato a retornar. */

auto ExtraccionData::Promedio(Eigen::MatrixXd datos) ->
decltype (datos.colwise().mean()){
    return datos.colwise().mean();
}

/* Para implementar la función de desviación estandar
 * datos = xi - x.promedio()
 * En esta función */

auto ExtraccionData::DesvStandard(Eigen::MatrixXd datos)-> decltype (((datos.array().square().colwise().sum())/(datos.rows()-1)).sqrt()){

    return ((datos.array().square().colwise().sum())/(datos.rows()-1)).sqrt();
}
/*Se implementa la función que calcule la normalización
 * de los datos, lo anterior para regular la escala o
 * magnitud de los datos, por lo tanto asegurar la
 * precisición de los modelos de machine learning */

Eigen::MatrixXd ExtraccionData::Normalizador(Eigen::MatrixXd datos){
    Eigen::MatrixXd datosEsc = datos.rowwise()-Promedio(datos);

    Eigen::MatrixXd NormMatriz = datosEsc.array().rowwise()/DesvStandard(datosEsc);

    return NormMatriz;
}

/* A continuación se implementa la función para hacer la
 * división de datos en dos grupos. El primer grupo es para
 * entrenamiento, por lo general se usa del 70% al 80% del
 * total de los datos. El segundo grupo de datos, es para
 * pruebas. Se requiere crear una función que devuelva dos
 * grupos de datos, seleccionands de forma aleatoria */

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> ExtraccionData::TrainTestDiv(Eigen::MatrixXd datos, float sizeTrain){
   int filas = datos.rows();
   int filasTrain = round(sizeTrain*filas);
   int filasTest = filas-filasTrain;

/* Con Eigen, se puede especificar un bloque de una matrix,
 * seleccionando las filas superiores para el conjunto de
 * entrenamiento, y las demás para el conjunto de pruebas */

   Eigen::MatrixXd trainMatriz = datos.topRows(filasTrain);
   /* Una vez seleccionas las filas superiores para entrenamiento
    * se seleccionan las columnas a la izquierda (OJO/WARNING:
    * para este conjunto de datos) correspondiente a las
    * "features" o variables dependientes:
    * Entones se seleccionan la cantidad de columnas - 1. */
   Eigen::MatrixXd X_train = trainMatriz.leftCols(datos.cols()-1);
   /* se selecciona la variable dependiente, en los datos de
    * entrenamiento */
   Eigen::MatrixXd y_train = trainMatriz.rightCols(1);

   /*se realiza el mismo procedimiento para el conjunto de datos
    * de prueba, recordando que se tiene los datos de la parte
    * inferior de la matriz de entrad, la función bottomRows
    * devuelve la parte inferior de la matriz */

   Eigen::MatrixXd testMatriz = datos.bottomRows(filasTest);
   /* Una vez seleccionas las filas superiores para entrenamiento
    * se seleccionan las columnas a la izquierda (OJO/WARNING:
    * para este conjunto de datos) correspondiente a las
    * "features" o variables dependientes:
    * Entones se seleccionan la cantidad de columnas - 1. */
   Eigen::MatrixXd X_test = testMatriz.leftCols(datos.cols()-1);
   /* se selecciona la variable dependiente, en los datos de
    * entrenamiento */
   Eigen::MatrixXd y_test = testMatriz.rightCols(1);
   /* Finalmente se retorna la tupla, que contiene
    * los subconjuntos de prueba y entrenamiento
    * Atención con la tupla enviada, dado que al ser usada
    * es necesario desempaquetarla */

   return std::make_tuple(X_train, y_train, X_test, y_test);
}
 /* a continuación se crea una funcion para exportar los valores
  * de vector a archivo */
void ExtraccionData::vectorToFile(std::vector<float> vectorData, std::string nameFile){
    /*Se crea la salida de datos stream o flujo de datos, del fichero de entrada*/
    std::ofstream salidaData(nameFile);
    /* Se escribe cada objeto del tipo float sobre data vector,
     * condicionado por un cambio de linea*/
    std::ostream_iterator<float> Datavector(salidaData, "\n");
    /* se hace una copia de  los objetos escritos sobre el vectordata*/
    std::copy(vectorData.begin(), vectorData.end(), Datavector);

}
/* a continuación se implementa una función para exportar una matriz
 * dinamica double a fichero.
 * Las exportaciones a ficheros son criticas o significativas
 * en tanto se tiene seguridad trazabilidad y control
 * sobre los resultados parciales obtenidos */

void ExtraccionData::matrixToFile(Eigen::MatrixXd DataMatrix, std::string nameFile){
    /*Se crea la salida de datos stream o flujo de datos, del fichero de entrada*/
    std::ofstream salidaData(nameFile);
    /*Si el fichero esta abierto, y no ha llegado al final
     * copie los datos de la matrix */
    if(salidaData.is_open()){
        salidaData << DataMatrix << "\n";

    }

}





/* WARNING: **********ADVERTENCIA***********
 * se ha de estudiar los datos para saber las posiciones
 * sobre las columnas: variable dependiente/variables Independientes */



















