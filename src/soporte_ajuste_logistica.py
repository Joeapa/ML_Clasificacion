# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np

# Visualizaciones
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree

# Para realizar la clasificación y la evaluación del modelo
# -----------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb

# Para realizar cross validation
# -----------------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.preprocessing import KBinsDiscretizer


class AnalisisModelosClasificacion:
    def __init__(self, dataframe, variable_dependiente):
        self.dataframe = dataframe
        self.variable_dependiente = variable_dependiente
        self.X = dataframe.drop(variable_dependiente, axis=1)
        self.y = dataframe[variable_dependiente]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, train_size=0.8, random_state=42, shuffle=True
        )

        # Modelos
        self.model_logistic_regression = LogisticRegression()
        self.model_tree = DecisionTreeClassifier()
        self.model_random_forest = RandomForestClassifier()
        self.model_gradient_boosting = GradientBoostingClassifier()
        self.model_xgboost = xgb.XGBClassifier()

        # Resultados de los modelos
        self.y_predict_train_lr = None
        self.y_predict_test_lr = None
        self.y_predict_train_tree = None
        self.y_predict_test_tree = None
        self.y_predict_train_rf = None
        self.y_predict_test_rf = None
        self.y_predict_train_gb = None
        self.y_predict_test_gb = None
        self.y_predict_train_xgb = None
        self.y_predict_test_xgb = None
        self.best_model_lr = None
        self.best_model_tree = None
        self.best_model_rf = None
        self.best_model_gb = None
        self.best_model_xgb = None



    def ajustar_modelo_logistic_regression(self, param_grid=None):
        """
        Ajusta el modelo de regresión logística.
        """
        if param_grid is None:
            param_grid = {
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'saga'],
                'max_iter': [100, 200, 500]
            }

        grid_search = GridSearchCV(estimator=self.model_logistic_regression, param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        self.best_model_lr = grid_search.best_estimator_
        self.y_predict_train_lr = self.best_model_lr.predict(self.X_train)
        self.y_predict_test_lr = self.best_model_lr.predict(self.X_test)



    def ajustar_modelo_tree(self, param_grid=None):
        if param_grid is None:
            param_grid = {
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        grid_search = GridSearchCV(estimator=self.model_tree, param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        self.best_model_tree = grid_search.best_estimator_
        self.y_predict_train_tree = self.best_model_tree.predict(self.X_train)
        self.y_predict_test_tree = self.best_model_tree.predict(self.X_test)

    def ajustar_modelo_random_forest(self, param_grid=None):
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            }
        
        grid_search = GridSearchCV(estimator=self.model_random_forest, param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        self.best_model_rf = grid_search.best_estimator_
        self.y_predict_train_rf = self.best_model_rf.predict(self.X_train)
        self.y_predict_test_rf = self.best_model_rf.predict(self.X_test)

    def ajustar_modelo_gradient_boosting(self, param_grid=None):
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 1.0]
            }
        
        grid_search = GridSearchCV(estimator=self.model_gradient_boosting, param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        self.best_model_gb = grid_search.best_estimator_
        self.y_predict_train_gb = self.best_model_gb.predict(self.X_train)
        self.y_predict_test_gb = self.best_model_gb.predict(self.X_test)

    def ajustar_modelo_xgboost(self, param_grid=None):
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        
        grid_search = GridSearchCV(estimator=self.model_xgboost, param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        self.best_model_xgb = grid_search.best_estimator_
        self.y_predict_train_xgb = self.best_model_xgb.predict(self.X_train)
        self.y_predict_test_xgb = self.best_model_xgb.predict(self.X_test)

    def obtener_resultados(self, modelo):
        """
        Obtiene los resultados de las predicciones para cualquier modelo ajustado.
        
        Args:
        - modelo: El modelo ajustado para el cual se quieren obtener los resultados ('logistic_regression', 'tree', 'random_forest', 'gradient_boosting', 'xgboost').
        
        Returns:
        Un DataFrame con los resultados de predicciones para el conjunto de entrenamiento y prueba.
        """
        def crear_resultados(real, predicho, conjunto, modelo_nombre):
            return pd.DataFrame({
                'Real': real,
                'Predicho': predicho,
                'Conjunto': conjunto,
                'Modelo': modelo_nombre
            })

        if modelo == 'logistic_regression':
            if self.y_predict_train_lr is None or self.y_predict_test_lr is None:
                raise ValueError("Debe ajustar el modelo de regresión logística antes de obtener resultados.")
            y_train_pred, y_test_pred = self.y_predict_train_lr, self.y_predict_test_lr
            model_name = "Regresión Logística"
        elif modelo == 'tree':
            if self.y_predict_train_tree is None or self.y_predict_test_tree is None:
                raise ValueError("Debe ajustar el modelo de árbol de decisión antes de obtener resultados.")
            y_train_pred, y_test_pred = self.y_predict_train_tree, self.y_predict_test_tree
            model_name = "Árbol de Decisión"
        elif modelo == 'random_forest':
            if self.y_predict_train_rf is None or self.y_predict_test_rf is None:
                raise ValueError("Debe ajustar el modelo Random Forest antes de obtener resultados.")
            y_train_pred, y_test_pred = self.y_predict_train_rf, self.y_predict_test_rf
            model_name = "Random Forest"
        elif modelo == 'gradient_boosting':
            if self.y_predict_train_gb is None or self.y_predict_test_gb is None:
                raise ValueError("Debe ajustar el modelo Gradient Boosting antes de obtener resultados.")
            y_train_pred, y_test_pred = self.y_predict_train_gb, self.y_predict_test_gb
            model_name = "Gradient Boosting"
        elif modelo == 'xgboost':
            if self.y_predict_train_xgb is None or self.y_predict_test_xgb is None:
                raise ValueError("Debe ajustar el modelo XGBoost antes de obtener resultados.")
            y_train_pred, y_test_pred = self.y_predict_train_xgb, self.y_predict_test_xgb
            model_name = "XGBoost"
        else:
            raise ValueError("Modelo no reconocido. Debe ser 'logistic_regression', 'tree', 'random_forest', 'gradient_boosting' o 'xgboost'.")

        # Crear los DataFrames de resultados para entrenamiento y prueba
        resultados_train = crear_resultados(self.y_train, y_train_pred, 'Train', model_name)
        resultados_test = crear_resultados(self.y_test, y_test_pred, 'Test', model_name)

        # Concatenar resultados y devolverlos
        resultados = pd.concat([resultados_train, resultados_test], axis=0)
        
        return resultados
    
    def plot_matriz_confusion(self, modelo):
        """
        Calcula y plotea la matriz de confusión para el modelo seleccionado.
        
        Args:
        - modelo: el modelo para el cual se va a calcular la matriz de confusión ('logistic_regression', 'tree', 'random_forest', 'gradient_boosting', 'xgboost').
        - labels: Lista opcional de etiquetas de clases para usar en el plot.
        """
        if modelo == 'logistic_regression':
            if self.y_predict_test_lr is None:
                raise ValueError("Debe ajustar el modelo de regresión logística antes de calcular la matriz de confusión.")
            y_true, y_pred = self.y_test, self.y_predict_test_lr
        elif modelo == 'tree':
            if self.y_predict_test_tree is None:
                raise ValueError("Debe ajustar el modelo de árbol de decisión antes de calcular la matriz de confusión.")
            y_true, y_pred = self.y_test, self.y_predict_test_tree
        elif modelo == 'random_forest':
            if self.y_predict_test_rf is None:
                raise ValueError("Debe ajustar el modelo Random Forest antes de calcular la matriz de confusión.")
            y_true, y_pred = self.y_test, self.y_predict_test_rf
        elif modelo == 'gradient_boosting':
            if self.y_predict_test_gb is None:
                raise ValueError("Debe ajustar el modelo Gradient Boosting antes de calcular la matriz de confusión.")
            y_true, y_pred = self.y_test, self.y_predict_test_gb
        elif modelo == 'xgboost':
            if self.y_predict_test_xgb is None:
                raise ValueError("Debe ajustar el modelo XGBoost antes de calcular la matriz de confusión.")
            y_true, y_pred = self.y_test, self.y_predict_test_xgb
        else:
            raise ValueError("Modelo no reconocido. Debe ser 'logistic_regression', 'tree', 'random_forest', 'gradient_boosting' o 'xgboost'.")

        # Calcular la matriz de confusión
        matriz_conf = confusion_matrix(y_true, y_pred)

        # Ploteo de la matriz de confusión
        plt.figure(figsize=(8, 6))
        sns.heatmap(matriz_conf, annot=True, fmt='g', cmap='Blues')
        plt.title(f"Matriz de Confusión ({modelo})")
        plt.xlabel("Predicción")
        plt.ylabel("Valor Real")
        plt.show()

        
    def calcular_metricas(self, modelo):
        """
        Calcula las métricas de rendimiento para el modelo seleccionado.
        """
        if modelo == 'logistic_regression':
            if self.y_predict_train_lr is None or self.y_predict_test_lr is None:
                raise ValueError("Debe ajustar el modelo de regresión logística antes de calcular las métricas.")
            y_train_pred, y_test_pred = self.y_predict_train_lr, self.y_predict_test_lr
        elif modelo == 'tree':
            if self.y_predict_train_tree is None or self.y_predict_test_tree is None:
                raise ValueError("Debe ajustar el modelo de árbol de decisión antes de calcular las métricas.")
            y_train_pred, y_test_pred = self.y_predict_train_tree, self.y_predict_test_tree
        elif modelo == 'random_forest':
            if self.y_predict_train_rf is None or self.y_predict_test_rf is None:
                raise ValueError("Debe ajustar el modelo Random Forest antes de calcular las métricas.")
            y_train_pred, y_test_pred = self.y_predict_train_rf, self.y_predict_test_rf
        elif modelo == 'gradient_boosting':
            if self.y_predict_train_gb is None or self.y_predict_test_gb is None:
                raise ValueError("Debe ajustar el modelo Gradient Boosting antes de calcular las métricas.")
            y_train_pred, y_test_pred = self.y_predict_train_gb, self.y_predict_test_gb
        elif modelo == 'xgboost':
            if self.y_predict_train_xgb is None or self.y_predict_test_xgb is None:
                raise ValueError("Debe ajustar el modelo XGBoost antes de calcular las métricas.")
            y_train_pred, y_test_pred = self.y_predict_train_xgb, self.y_predict_test_xgb
        else:
            raise ValueError("Modelo no reconocido. Debe ser 'tree', 'random_forest', 'gradient_boosting' o 'xgboost'.")

        # Calcular métricas para el conjunto de entrenamiento
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        train_precision = precision_score(self.y_train, y_train_pred, average='weighted', zero_division=0)
        train_recall = recall_score(self.y_train, y_train_pred, average='weighted', zero_division=0)
        train_f1 = f1_score(self.y_train, y_train_pred, average='weighted', zero_division=0)

        # Calcular métricas para el conjunto de prueba
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        test_precision = precision_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
        test_recall = recall_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(self.y_test, y_test_pred, average='weighted', zero_division=0)

        # Crear un DataFrame con las métricas
        metricas = pd.DataFrame({
            'train': {
                'accuracy': train_accuracy,
                'precision': train_precision,
                'recall': train_recall,
                'f1_score': train_f1
            },
            'test': {
                'accuracy': test_accuracy,
                'precision': test_precision,
                'recall': test_recall,
                'f1_score': test_f1
            }
        })

        return metricas
    
    def plot_arbol_decision(self):
        """
        Plotea el árbol de decisión entrenado.
        """
        if self.best_model_tree is None:
            raise ValueError("Debe ajustar el modelo de árbol de decisión antes de plotear.")

        fig = plt.figure(figsize=(40, 20))
        tree.plot_tree(self.best_model_tree, feature_names=self.X_train.columns, filled=True)
        plt.show()


    def importancia_predictores(self, modelo):
        """
        Muestra la importancia de los predictores para el modelo seleccionado.
        """
        if modelo not in ['logistic_regression', 'tree', 'random_forest', 'gradient_boosting', 'xgboost']:
            raise ValueError("Modelo no reconocido. Debe ser 'logistic_regression', 'tree', 'random_forest', 'gradient_boosting' o 'xgboost'.")

        if modelo == 'logistic_regression':
            if self.best_model_lr is None:
                raise ValueError("Debe ajustar el modelo de regresión logística antes de obtener la importancia de los predictores.")
            
            # Los coeficientes de regresión logística representan la importancia
            importancias = np.abs(self.best_model_lr.coef_[0])  # Valor absoluto de los coeficientes
            model_name = "Regresión Logística"

        elif modelo == 'tree':
            if self.best_model_tree is None:
                raise ValueError("Debe ajustar el modelo de árbol de decisión antes de obtener la importancia de los predictores.")
            importancias = self.best_model_tree.feature_importances_
            model_name = "Árbol de Decisión"

        elif modelo == 'random_forest':
            if self.best_model_rf is None:
                raise ValueError("Debe ajustar el modelo de Random Forest antes de obtener la importancia de los predictores.")
            importancias = self.best_model_rf.feature_importances_
            model_name = "Random Forest"

        elif modelo == 'gradient_boosting':
            if self.best_model_gb is None:
                raise ValueError("Debe ajustar el modelo Gradient Boosting antes de obtener la importancia de los predictores.")
            importancias = self.best_model_gb.feature_importances_
            model_name = "Gradient Boosting"

        elif modelo == 'xgboost':
            if self.best_model_xgb is None:
                raise ValueError("Debe ajustar el modelo XGBoost antes de obtener la importancia de los predictores.")
            importancias = self.best_model_xgb.feature_importances_
            model_name = "XGBoost"

        # Crear un DataFrame con las importancias
        importancia_predictores = pd.DataFrame(
            {'predictor': self.X_train.columns,
             'importancia': importancias}
        )

        importancia_predictores.sort_values(by=["importancia"], ascending=False, inplace=True)

        print(f"Importancia de los predictores en el modelo {model_name}")
        print("-------------------------------------------")
        print(importancia_predictores)

        # Ploteamos los resultados
        plt.figure(figsize=(10, 6))
        sns.barplot(x="importancia", y="predictor", data=importancia_predictores, palette="viridis")
        plt.show()


    def cross_validate_kfold(self, modelo, cv=5, scoring='accuracy'):
        """
        Aplica K-Fold Cross-Validation al modelo seleccionado.
        """
        if modelo == "logistic_regression":
            estimator = self.model_logistic_regression
        elif modelo == 'tree':
            estimator = self.model_tree
        elif modelo == 'random_forest':
            estimator = self.model_random_forest
        elif modelo == 'gradient_boosting':
            estimator = self.model_gradient_boosting
        elif modelo == 'xgboost':
            estimator = self.model_xgboost
        else:
            raise ValueError("Modelo no reconocido. Debe ser 'tree', 'random_forest', 'gradient_boosting' o 'xgboost'.")

        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(estimator, self.X, self.y, cv=kf, scoring=scoring)

        print(f"K-Fold Cross-Validation Scores para un modelo de {modelo} es:")
        print(f"Mean Score: {round(abs(scores.mean()),2)}")


    def cross_validate_stratified_kfold(self, modelo, cv=5, scoring='accuracy', n_bins=5):
        """
        Aplica Stratified K-Fold Cross-Validation al modelo seleccionado.
        Nota: Este método estratifica la variable dependiente en 'n_bins' categorías.
        """
        if modelo == "logistic_regression":
            estimator = self.model_logistic_regression
        elif modelo == 'tree':
            estimator = self.model_tree
        elif modelo == 'random_forest':
            estimator = self.model_random_forest
        elif modelo == 'gradient_boosting':
            estimator = self.model_gradient_boosting
        elif modelo == 'xgboost':
            estimator = self.model_xgboost
        else:
            raise ValueError("Modelo no reconocido. Debe ser 'tree', 'random_forest', 'gradient_boosting' o 'xgboost'.")

        # Estratificación de la variable dependiente
        y_binned = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform').fit_transform(self.y.values.reshape(-1, 1))

        # Aplicar Stratified K-Fold Cross-Validation
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(estimator, self.X, self.y, cv=skf.split(self.X, y_binned.ravel()), scoring=scoring)

        print(f"Stratified K-Fold Cross-Validation Scores para un modelo {modelo} es:")
        print(f"Mean Score: {round(abs(scores.mean()),2)}")
    
    def cross_validate_stratified_kfold(self, modelo, cv=5, scoring='accuracy'):
        """
        Aplica Stratified K-Fold Cross-Validation al modelo seleccionado.
        
        Args:
        - modelo: El modelo sobre el cual se realizará la validación cruzada ('logistic_regression', 'tree', 'random_forest', 'gradient_boosting', 'xgboost').
        - cv: El número de particiones (folds) que se usarán en la validación cruzada.
        - scoring: La métrica de evaluación, por defecto 'accuracy'.
        """
        # Selección del modelo a evaluar
        if modelo == "logistic_regression":
            estimator = self.model_logistic_regression
        elif modelo == 'tree':
            estimator = self.model_tree
        elif modelo == 'random_forest':
            estimator = self.model_random_forest
        elif modelo == 'gradient_boosting':
            estimator = self.model_gradient_boosting
        elif modelo == 'xgboost':
            estimator = self.model_xgboost
        else:
            raise ValueError("Modelo no reconocido. Debe ser 'logistic_regression', 'tree', 'random_forest', 'gradient_boosting' o 'xgboost'.")

        # Aplicar Stratified K-Fold Cross-Validation
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(estimator, self.X, self.y, cv=skf, scoring=scoring)

        print(f"Stratified K-Fold Cross-Validation Scores para un modelo de {modelo} es:")
        print(f"Mean Score: {round(abs(scores.mean()), 2)}")