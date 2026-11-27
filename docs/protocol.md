cd /mnt/c/Users/DELL/openseg-r
pwd    # doit afficher /mnt/c/Users/DELL/openseg-r
# Protocole d'évaluation (v1)

Ce document est écrit **avant** de lancer les évaluations. Il définit ce qui est mesuré
et comment. Toute modification ultérieure est datée et justifiée.

## 1. Tâche évaluée

Segmentation open-vocabulary : à partir d'une image et d'un texte (le nom d'un groupe),
le système prédit les masques des objets correspondants.
Pipeline : Grounding DINO (texte -> boîtes) puis SAM 2 (boîtes -> masques), sans entraînement.

## 2. Données

- TACO, annotations officielles (`annotations.json`), 1500 images, 60 catégories.
- Quatre groupes (`configs/categories.json`) : bottle, can, bag, cigarette.
  Le groupe « carton » est écarté (catégories hétérogènes).
- **Unité d'évaluation : la paire (image, groupe).** Le texte est celui du groupe.
  Les références sont les seuls objets de ce groupe dans l'image.
- Découpage (`configs/split_v1.json`, graine 0) : dev = 20 paires par groupe,
  test = 40 paires par groupe. Une image n'apparaît que dans une seule paire.
- Les images sont lues avec correction de l'orientation EXIF (`openseg.data.load_image`).

## 3. Métriques

### 3.1 Métrique principale : IoU « union » par paire
Tous les masques prédits de l'image sont fusionnés en un masque binaire, de même pour
les références. IoU = intersection / union.
- Une paire **sans aucune prédiction** a un IoU de **0** : elle n'est jamais retirée.
- Moyenne par groupe, puis **macro-moyenne sur les 4 groupes** (chaque groupe pèse autant).

### 3.2 Métriques d'accompagnement : appariement par instance
Matrice d'IoU entre masques prédits et masques de référence, appariement un-à-un
(algorithme hongrois). Un appariement est valide si IoU >= 0,5.
- Rappel = tp / (tp + fn) ; précision = tp / (tp + fp) ; F1 = 2tp / (2tp + fp + fn).
  Agrégées par groupe en sommant tp, fp, fn sur les paires.
- IoU moyen des paires valides : qualité du découpage, indépendamment de la détection.

### 3.3 Incertitude
Intervalle de confiance à 95 % par bootstrap (2000 ré-échantillonnages des paires,
dans chaque groupe). Une différence entre variantes n'est commentée que si elle dépasse
l'incertitude.

## 4. Expériences

- **Pipeline complet** : Grounding DINO + SAM 2.
- **Oracle** : SAM 2 reçoit les boîtes de référence de TACO. Cela mesure la limite de
  SAM 2 seul. L'écart entre l'oracle et le pipeline complet mesure ce que perd le détecteur.

## 5. Discipline expérimentale

- Le texte, le seuil et les variantes sont choisis sur **dev**.
- **Test** : une seule exécution finale, après figement de la configuration.
- Chaque expérience répond à une question précise et se termine par une conclusion
  soutenue par les chiffres et leur intervalle de confiance.

## 6. Limites connues (à rappeler dans le rapport)

- Objets non annotés dans TACO (dont la catégorie « Unlabeled litter ») : des détections
  correctes peuvent être comptées comme faux positifs, ce qui baisse la précision.
- « Plastic film » et « Other plastic wrapper » ne sont pas dans le groupe bag : un film
  détecté comme « plastic bag » est un faux positif.
- Le groupe bottle inclut les bouteilles en verre alors que le texte de base est « bottle ».
- Petits effectifs (40 paires par groupe en test) : les écarts faibles sont incertains.
- Le seuil d'appariement de 0,5 est un choix conventionnel.
