#include "regresion_lineal.h"
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <stdio.h>

/* Se necesita entrenar el modelo, lo que implica
 * minimizar la función de costo. De esta forma se
 * puede medir la función de hipotesis. Una función
 * de costo es la forma de penalizar al modelo por
 * cometer un error.
 * Se implementa una función de tipo flotante que
 * toma como entrada los valores de (X,y) */

float RegresionLineal::OLS_costo(Eigen::MatrixXd X,Eigen::MatrixXd y,Eigen::MatrixXd Theta){
    Eigen::MatrixXd Diferencia = pow((X*Theta - y).array(),2);
    return (Diferencia.sum()/(2*X.rows()));
}
/* Se provee al programa una función para dar al
 * algoritmo un valor inicial, el cual cambiara
 * iterativamente hasta que converja al valor minimo
 * de la función de costo. Basicamente describe el Gradiente
 * Desendiente: La idea es calvular el gradiente para la función
 * de costo, dado por la derivada parcial de la función.
 * La funcion debe tener un Alfa que representa el salto del gradiente.
 * Las entradas para la función son x, y, theta, Alfa y el numero de
 * iteraciones */

std::tuple<Eigen::VectorXd, std::vector<float>> RegresionLineal::Gradiente(Eigen::MatrixXd X,Eigen::MatrixXd y,Eigen::VectorXd Theta, float alfa, int iterator){

    /* Se almacena parametros de theta*/
    Eigen::MatrixXd temporal = Theta;

    /* Secapturan el numero de variables independientes*/
    int parametros = Theta.rows();

    /* Se hubica el costo inicial, que se actualiza cada vez
     * con los nuevos pesos(pendientes)*/
    std::vector<float> costo;
    costo.push_back(OLS_costo(X, y, Theta));

    /* Por cada iteración se calcula la función de error de cada
     * variable independiente para ser almacenado en la variable
     * temporal(tempTheta) basada en el nuevo valor de Theta */

    for(int i=0; i<iterator; ++i){
        Eigen::MatrixXd error = X*Theta - y ;
        for(int j=0; j<parametros; ++j){
            Eigen::MatrixXd X_i = X.col(j);
            Eigen::MatrixXd tempTheta = error.cwiseProduct(X_i);
            temporal(j,0) = Theta(j,0) - ((alfa/X.rows())*tempTheta.sum());
        }
        Theta = temporal;
        costo.push_back(OLS_costo(X, y, Theta));
    }
    /* Se empaqueta la upla y retonamos*/
    return std::make_tuple(Theta, costo);

}
 /*para determinar que tan bueno nuestro modelo es necesario acudir a una metrica
  * de rendimiento para ellos se escoge el R2 el cual representa que tan bueno es
  * nuestro modelo*/
float RegresionLineal::R2(Eigen::MatrixXd y,Eigen::MatrixXd y_hat){
    auto Numerador = pow((y-y_hat).array(),2).sum();
    auto Denominador = pow(y.array()-y.mean(),2).sum();

    return 1-Numerador/Denominador;
}



