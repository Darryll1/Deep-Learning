**Prédiction de Prix de Maisons à King County, USA**

Projet réalisé dans le cadre du TP de Deep Learning portant sur l'analyse de données, la comparaison de modèles et le développement d'une application interactive.

Le projet est divisé en deux parties :
- Notebook d'entraînement : entraînement et comparaison de plusieurs modèles de régression.
- Application Streamlit : exploration des données et prédiction de prix en ligne.

**Objectifs**

- Réaliser une analyse exploratoire des données.
- Construire plusieurs modèles de régression :
  Random Forest Regressor (Scikit-Learn)
  TabNet (PyTorch Tabular)
  FTTransformer (PyTorch Tabular)
- Comparer les modèles selon trois métriques :
  RMSE (Root Mean Squared Error)
  MAE (Mean Absolute Error)
  R² (Coefficient de détermination)
- Sélectionner automatiquement le meilleur modèle.
- Développer une application Streamlit interactive :
- Exploration des données avant et après transformations.
- Prédiction de prix à partir de caractéristiques saisies.

  **Sources**

- Dataset : House Sales in King County (Kaggle)
- Libraries : Scikit-Learn, PyTorch Tabular, Streamlit
  
  **Fichiers du projet**

| Fichier                               | Description                                                                                       |
| :------------------------------------ | :------------------------------------------------------------------------------------------------ |
| `Prédiction_Prix_Maison_USA-v2.ipynb` | Notebook complet d'analyse, entraînement, évaluation et sauvegarde automatique du meilleur modèle |
| `app-deep.py`                         | Application web développée avec Streamlit pour explorer les données et faire des prédictions      |
| `kc_house_data.csv`                   | Jeu de données brut utilisé pour l'entraînement                                                   |
| `dataframe-transformée.csv`           | Jeu de données transformé pour amélioration des performances                                      |
| `random_forest_model.joblib`          | Meilleur modèle sauvegardé prêt à l'emploi                                                        |

**Application Streamlit**

- Exploration :
  Affichage des données avant et après transformations.
  Statistiques descriptives claires et accessibles.
- Prédiction :
  Interface utilisateur pour saisir les caractéristiques d'une maison.
  Prédiction du prix affichée instantanément.
  
**Installation et Lancement**

1. Prérequis
pip install pandas numpy scikit-learn streamlit joblib torch pytorch-tabular

2. Entraînement des modèles
Lancez le notebook :
jupyter notebook Prédiction_Prix_Maison_USA-v2.ipynb

3. Lancement de l'Application
streamlit run app-deep.py

**Justification des Modèles**

- Random Forest Regressor : 
Méthode robuste et rapide pour des données tabulaires.
Gestion des variables non-linéaires sans normalisation.

- TabNet : 
Deep learning spécialisé pour des structures tabulaires.
Apprentissage automatique des masques de features.

- FTTransformer : 
Version Transformer optimisée pour des données tabulaires.
Apprentissage des dépendances non linéaires complexes.


