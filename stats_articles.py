import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

########################################################################################################################
########################################################################################################################
################################## STATS ARTICLES JOURNAUX #############################################################
########################################################################################################################
########################################################################################################################

### import données

# donnees epa 2009-2018
data_epa = pd.read_excel("G:/data_hugo/doctorat/donnees_BF/vars_expl_sa/code_camille/resultats_traites/epa_val.xlsx")

# sorties journaux lefaso et burk24
data_burk24 = pd.read_csv("G:/data_hugo/doctorat/donnees_BF/vars_expl_sa/code_camille/resultats_traites/articles_burk24.csv", sep=";")
data_lefaso = pd.read_csv("G:/data_hugo/doctorat/donnees_BF/vars_expl_sa/code_camille/resultats_traites/articles_lefaso.csv", sep=";")

data_journ = pd.concat([data_burk24, data_lefaso], axis=0).reset_index().drop(["index"], axis=1)

# voc SA
voc_det_sa_src = open("G:/data_hugo/doctorat/donnees_BF/vars_expl_sa/code_camille/traitements_identification_articles/lexique_dictionnaire/lexique_detaille_sa.txt", "r", encoding="utf-8")
voc_det_sa = voc_det_sa_src.read()
voc_det_sa = list(voc_det_sa.split("\n"))

# voc Crises
voc_det_cr_src = open("G:/data_hugo/doctorat/donnees_BF/vars_expl_sa/code_camille/traitements_identification_articles/lexique_dictionnaire/lexique_detaille_crise.txt", "r", encoding="utf-8")
voc_det_cr = voc_det_cr_src.read()
voc_det_cr = list(voc_det_cr.split("\n"))

# black word cloud
def black_color_func(word, font_size, position,orientation,random_state=None, **kwargs):
    return("hsl(0,100%, 1%)")

# seuil x
x = 0.36

print("Nombre d'articles", data_journ.shape[0])

################################################# analyse globale #######################################################################
print("Analyse globale")

## proportion articles SA
print("Proportion articles SA", data_journ[data_journ.SIM_W2V>x].shape[0]/data_journ.shape[0]*100)

## négativité moyenne des articles qui traitent SA
print("Proportion d'articles SA négatifs", data_journ[(data_journ.SIM_W2V>x) & (data_journ.NEG>0.1)].shape[0]/data_journ[data_journ.SIM_W2V>x].shape[0]*100)

## top 10 vocabulaire détaillé SA
dict_occur = {}
for mot in voc_det_sa:
    nb_mots = (data_journ[data_journ.SIM_W2V>x].VOC_SA.str.lower().str.count(mot.lower())/data_journ[data_journ.SIM_W2V>x].NB_MOTS).mean()
    dict_occur[mot] = round(nb_mots, 6)
top10 = dict(Counter(dict_occur).most_common(30))
data_df = pd.DataFrame(top10.items(), columns=['MOT', 'FREQUENCE'])
#data_df['Exps'] = data_df['MOT'] + " (" + data_df['FREQUENCE'].astype(str) + ")"
print("Top5 voc SA\n", data_df['MOT'].head())
data_df['MOT'].to_excel(excel_writer="result_global/top_10_global_sa.xlsx", index=False)

data_cloud = data_df.set_index('MOT').to_dict()['FREQUENCE']
wordcloud = WordCloud(background_color="white", width=3000, height=2000).generate_from_frequencies(data_cloud)
wordcloud.recolor(color_func = black_color_func)
plt.axis("off")
plt.imshow(wordcloud, interpolation="bilinear")
plt.savefig('result_global/Wordcloud_global_SA.png')

## top 10 vocabulaire détaillé Crises
dict_occur = {}
for mot in voc_det_cr:
    nb_mots = (data_journ[data_journ.SIM_W2V > x].VOC_CR.str.lower().str.count(mot.lower()) / data_journ[data_journ.SIM_W2V > x].NB_MOTS).mean()
    dict_occur[mot] = round(nb_mots, 6)
top10 = dict(Counter(dict_occur).most_common(30))
data_df = pd.DataFrame(top10.items(), columns=['MOT', 'FREQUENCE'])
# data_df['Exps'] = data_df['MOT'] + " (" + data_df['FREQUENCE'].astype(str) + ")"
print("Top5 voc CR\n", data_df['MOT'].head())
data_df['MOT'].to_excel(excel_writer="result_global/top_10_global_cr.xlsx",index=False)

data_cloud = data_df.set_index('MOT').to_dict()['FREQUENCE']
wordcloud = WordCloud(background_color="white", width=3000, height=2000).generate_from_frequencies(data_cloud)
wordcloud.recolor(color_func=black_color_func)
plt.axis("off")
plt.imshow(wordcloud, interpolation="bilinear")
plt.savefig('result_global/Wordcloud_global_CR.png')

##################################### analyse annuelle #################################################################
print("Analyse annuelle")

## proportion articles SA
prop_SA_annees = {}
for annee in range(2009, 2019):
    dat_annee = data_journ[data_journ.ANNEE == annee]
    prop_SA_annees[annee] = dat_annee[dat_annee.SIM_W2V > x].shape[0]/dat_annee.shape[0]*100
data_df = pd.DataFrame(prop_SA_annees.items(), columns=['ANNEE', 'PROP_SA'])
print("Proportion d'articles SA / annee\n", data_df)
data_df.to_excel(excel_writer="result_annee/prop_sa_annees.xlsx", index=False)

## négativité proportion
prop_SA_annees = {}
for annee in range(2009, 2019):
    dat_annee = data_journ[data_journ.ANNEE == annee]
    prop_SA_annees[annee] = dat_annee[(dat_annee.SIM_W2V > x) & (dat_annee.NEG>0.1)].shape[0]/ dat_annee[dat_annee.SIM_W2V > x].shape[0]*100
data_df = pd.DataFrame(prop_SA_annees.items(), columns=['ANNEE', 'PROP_NEG'])
print("Proportion d'articles SA négatifs / annee\n", data_df)
data_df.to_excel(excel_writer="result_annee/neg_prop_annees.xlsx", index=False)

## top 10 vocabulaire détaillé SA
data_tot = []
for annee in range(2009, 2019):
    dict_occur = {}
    dict_occur_global = {}
    for mot in voc_det_sa:
    #for mot in ["agriculture", "sécurité alimentaire", "riz", "pauvreté", "malnutrition", "pluie", "intrant", "foncier",
    #            "semence", "fruit", "élevage", "céréale", "maïs", "faim", "campagne agricole"]:
         tf = (data_journ[(data_journ.SIM_W2V>x) & (data_journ.ANNEE == annee)].VOC_SA.str.lower().str.count(mot.lower()) /
               data_journ[(data_journ.SIM_W2V > x) & (data_journ.ANNEE == annee)].NB_MOTS) # fréq relative de mot dans chaque article de l'année "annee"
         nbDocsMot = data_journ[(data_journ.SIM_W2V > x)].VOC_SA.str.lower().str.count(mot.lower())
         nbDocsMot = nbDocsMot[nbDocsMot>0].count() # nombre de documents qui parlent de SA, et tq mot y figure
         nbDocs = data_journ[(data_journ.SIM_W2V>x)].shape[0] # nombre de documents qui parlent de SA
         idf = np.log((1 + nbDocs) / (1 + nbDocsMot))
         tfidf = tf * idf
         tfidf_moy = tfidf.mean()
         dict_occur[mot] = round(tfidf_moy, 6)
    top10_ansa = dict(Counter(dict_occur).most_common(100))
    #top10_ansa = {k: v for k, v in sorted(top10_ansa.items(), key=lambda item: item[0])}
    data_df = pd.DataFrame(top10_ansa.items(), columns=['MOT', 'FREQUENCE'])
    nomvar = "Exps_" + str(annee)
    nomvar_f = "Freq_" + str(annee)
    data_df[nomvar] = data_df['MOT'] #+ " (" + data_df['FREQUENCE'].astype(str) + ")"
    data_df[nomvar_f] = data_df['FREQUENCE']
    data_tot.append(data_df[[nomvar, nomvar_f]])
    for mot in list(top10_ansa.keys()): # pour les 10 mots ayant le + grand tf-idf, calcul des infos pour obtenir ratio TIR
         tf_global = (data_journ[(data_journ.SIM_W2V>x) & (data_journ.ANNEE != annee)].VOC_SA.str.lower().str.count(mot.lower()) /
               data_journ[(data_journ.SIM_W2V > x) & (data_journ.ANNEE != annee)].NB_MOTS) # fréq relative de mot dans l'article de l'année "annee"
         nbDocsMot_global = data_journ[(data_journ.SIM_W2V > x)].VOC_SA.str.lower().str.count(mot.lower()) #
         nbDocsMot_global = nbDocsMot_global[nbDocsMot_global>0].count() # nombre de documents qui parlent de SA, et tq mot y figure
         nbDocs_global = data_journ[(data_journ.SIM_W2V>x)].shape[0] # nombre de documents qui parlent de SA
         idf_global = np.log((1 + nbDocs_global) / (1 + nbDocsMot_global))
         tfidf_global = tf_global * idf_global
         tfidf_global_moy = tfidf_global.mean()
         dict_occur_global[mot] = round(tfidf_global_moy, 6)

    # data to radar
    rows = [[top10_ansa[list(top10_ansa.keys())[i]] for i in range(10)], [dict_occur_global[list(top10_ansa.keys())[i]] for i in range(10)]]
    df_radar = pd.DataFrame(rows, columns=[list(top10_ansa.keys())[i] for i in range(10)], index=['tf-idf_' + str(annee), 'tf_id_other_years'])

    # ------- PART 1: Create background

    # number of variable
    categories = [list(top10_ansa.keys())[i] for i in range(10)]
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0, 2], ["0", "2"], color="black", size=7)
    plt.ylim(-1.5, 5)

    ax.xaxis.grid(True, color='grey', linestyle='--')
    ax.yaxis.grid(True, color='grey', linestyle=':')
    # ------- PART 2: Add plots

    # Plot each individual = each line of the data
    # I don't make a loop, because plotting more than 3 groups makes the chart unreadable

    # Ind1 (ratio TIR)
    values_ann = df_radar.iloc[0].values.flatten().tolist() #tf-idf moyen sur l'année "annee"
    values_oth = df_radar.iloc[1].values.flatten().tolist() #tf-idf moyen sur les autres années
    values = np.log([a / b for a, b in zip(values_ann, values_oth)]).tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="R - " + str(annee))
    ax.fill(angles, values, 'b', alpha=0.1)
    """
    # Ind1
    values = df_radar.iloc[0].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="tf-idf_" + str(annee))
    ax.fill(angles, values, 'b', alpha=0.1)
    
    # Ind2
    values = df_radar.iloc[1].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="tf_idf_oth_years", color="grey")
    ax.fill(angles, values, 'grey', alpha=0.1)
    """
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.2, 0.))
    # Show the graph
    #plt.show()
    plt.savefig('result_annee/radar/radar_sa_' + str(annee) + '.png')

    plt.clf()

data_tot = pd.concat(data_tot, axis=1)
print("Top5 voc SA ANNEES\n", data_tot.head())
data_tot.to_excel(excel_writer="result_annee/top_10_sa_annees.xlsx", index=False)

## top 10 vocabulaire détaillé CR
data_tot = []
for annee in range(2009, 2019):
    dict_occur = {}
    dict_occur_global = {}
    for mot in voc_det_cr:
    #for mot in ["catastrophe", "conflit", "pauvreté", "insécurité", "malnutrition", "inondation", "foncier", "faim",
    #            "migration", "crise alimentaire", "attaque", "sécheresse", "déplacement", "vulnérabilité", "tension"]:

         tf = (data_journ[(data_journ.SIM_W2V>x) & (data_journ.ANNEE == annee)].VOC_CR.str.lower().str.count(mot.lower()) /
               data_journ[(data_journ.SIM_W2V > x) & (data_journ.ANNEE == annee)].NB_MOTS) # fréq relative de mot dans l'article de l'année "annee"
         nbDocsMot = data_journ[(data_journ.SIM_W2V > x)].VOC_CR.str.lower().str.count(mot.lower())
         nbDocsMot = nbDocsMot[nbDocsMot>0].count() # nombre de documents qui parlent de SA, et tq mot y figure
         nbDocs = data_journ[(data_journ.SIM_W2V>x)].shape[0] # nombre de documents qui parlent de SA
         idf = np.log((1 + nbDocs) / (1 + nbDocsMot))
         tfidf = tf * idf
         tfidf_moy = tfidf.mean()
         dict_occur[mot] = round(tfidf_moy, 6)
    top10_ancr = dict(Counter(dict_occur).most_common(100))
    #top10_ancr = {k: v for k, v in sorted(top10_ancr.items(), key=lambda item: item[0])}
    data_df = pd.DataFrame(top10_ancr.items(), columns=['MOT', 'FREQUENCE'])
    nomvar = "Exps_" + str(annee)
    nomvar_f = "Freq_" + str(annee)
    data_df[nomvar] = data_df['MOT'] #+ " (" + data_df['FREQUENCE'].astype(str) + ")"
    data_df[nomvar_f] = data_df['FREQUENCE']
    data_tot.append(data_df[[nomvar, nomvar_f]])
    for mot in list(top10_ancr.keys()): # pour les 10 mots ayant le + grand tf-idf, calcul des infos pour obtenir ratio TIR
         tf_global = (data_journ[(data_journ.SIM_W2V>x) & (data_journ.ANNEE != annee)].VOC_CR.str.lower().str.count(mot.lower()) /
               data_journ[(data_journ.SIM_W2V > x) & (data_journ.ANNEE != annee)].NB_MOTS) # fréq relative de mot dans l'article de l'année "annee"
         nbDocsMot_global = data_journ[(data_journ.SIM_W2V > x)].VOC_CR.str.lower().str.count(mot.lower())
         nbDocsMot_global = nbDocsMot_global[nbDocsMot_global>0].count() # nombre de documents qui parlent de SA, et tq mot y figure
         nbDocs_global = data_journ[(data_journ.SIM_W2V>x)].shape[0] # nombre de documents qui parlent de SA
         idf_global = np.log((1 + nbDocs_global) / (1 + nbDocsMot_global))
         tfidf_global = tf_global * idf_global
         tfidf_global_moy = tfidf_global.mean()
         dict_occur_global[mot] = round(tfidf_global_moy, 6)
    # data to radar
    rows = [[top10_ancr[list(top10_ancr.keys())[i]] for i in range(10)], [dict_occur_global[list(top10_ancr.keys())[i]] for i in range(10)]]
    df_radar = pd.DataFrame(rows, columns=[list(top10_ancr.keys())[i] for i in range(10)],index=['tf-idf_' + str(annee), 'tf_id_other_years'])

    # ------- PART 1: Create background

    # number of variable
    categories = [list(top10_ancr.keys())[i] for i in range(10)]
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0, 2], ["0", "2"], color="black", size=7)
    plt.ylim(-1.5, 5)

    ax.xaxis.grid(True, color='grey', linestyle='--')
    ax.yaxis.grid(True, color='grey', linestyle=':')
    # ------- PART 2: Add plots

    # Plot each individual = each line of the data
    # I don't make a loop, because plotting more than 3 groups makes the chart unreadable
    # Ind1 (ratio TIR)
    values_ann = df_radar.iloc[0].values.flatten().tolist() #tf-idf moyen sur l'année "annee"
    values_oth = df_radar.iloc[1].values.flatten().tolist() #tf-idf moyen sur les autres années
    values = np.log([a / b for a, b in zip(values_ann, values_oth)]).tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="R - " + str(annee))
    ax.fill(angles, values, 'b', alpha=0.1)
    """
    # Ind1
    values = df_radar.iloc[0].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="tf-idf_" + str(annee))
    ax.fill(angles, values, 'b', alpha=0.1)
    
    # Ind2
    values = values = df_radar.iloc[1].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="tf_idf_oth_years", color="grey")
    ax.fill(angles, values, 'grey', alpha=0.1)
    """
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.2, 0.))

    # Show the graph
    #plt.show()
    plt.savefig('result_annee/radar/radar_cr_' + str(annee) + '.png')

    plt.clf()

data_tot = pd.concat(data_tot, axis=1)
print("Top5 voc CR ANNEES\n", data_tot.head())
data_tot.to_excel(excel_writer="result_annee/top_10_cr_annees.xlsx", index=False)
#################################### analyse régionale #################################################################
print("Analyse régionale")

col_reg = {'Centre':'REGION_CENTRE', 'Sahel':'REGION_SAHEL', 'Bassins':'REGION_BASSINS'}
for reg in ["Centre", "Sahel", "Bassins"]:
    print("\n" + reg)
    print("Nombre d'articles " + reg, data_journ[data_journ[col_reg[reg]] == reg].shape[0])

    ## proportion articles SA
    print("Proportion articles SA " + reg, data_journ[(data_journ.SIM_W2V>x) & (data_journ[col_reg[reg]] == reg)].shape[0]/data_journ[data_journ[col_reg[reg]] == reg].shape[0]*100)

    ## négativité moyenne des articles qui traitent SA
    print("proportion negativité articles SA " + reg, data_journ[(data_journ.SIM_W2V>x) & (data_journ.NEG>0.1) & (data_journ[col_reg[reg]] == reg)].shape[0]/data_journ[(data_journ.SIM_W2V>x) & (data_journ[col_reg[reg]] == reg)].shape[0]*100)

    ## top 10 vocabulaire détaillé SA
    dict_occur = {}
    for mot in voc_det_sa:
         tf = (data_journ[(data_journ.SIM_W2V>x) & (data_journ[col_reg[reg]] == reg)].VOC_SA.str.lower().str.count(mot.lower()) /
               data_journ[(data_journ.SIM_W2V > x) & (data_journ[col_reg[reg]] == reg)].NB_MOTS) # fréq relative de mot de chaque article de la région "region"
         nbDocsMot = data_journ[(data_journ.SIM_W2V > x)].VOC_SA.str.lower().str.count(mot.lower())
         nbDocsMot = nbDocsMot[nbDocsMot>0].count() # nombre de documents qui parlent de SA, et tq mot y figure
         nbDocs = data_journ[(data_journ.SIM_W2V>x)].shape[0] # nombre de documents qui parlent de SA,
         idf = np.log((1 + nbDocs) / (1 + nbDocsMot))
         tfidf = tf * idf
         tfidf_moy = tfidf.mean()
         dict_occur[mot] = round(tfidf_moy, 6)
    top10 = dict(Counter(dict_occur).most_common(100))
    data_df = pd.DataFrame(top10.items(), columns=['MOT', 'FREQUENCE'])
    # data_df['Exps'] = data_df['MOT'] + " (" + data_df['FREQUENCE'].astype(str) + ")"
    print("Top5 voc SA\n", data_df['MOT'].head())
    data_df['MOT'].to_excel(excel_writer="result_region/top_10_sa_" + reg + ".xlsx", index=False)
    data_cloud = data_df.set_index('MOT').to_dict()['FREQUENCE']
    wordcloud = WordCloud(background_color="white", width=3000, height=2000).generate_from_frequencies(data_cloud)
    wordcloud.recolor(color_func=black_color_func)
    plt.axis("off")
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.savefig('result_region/Wordcloud_' + reg + '_SA.png')

    ## top 10 vocabulaire détaillé Crises
    dict_occur = {}
    for mot in voc_det_cr:
        tf = (data_journ[(data_journ.SIM_W2V > x) & (data_journ[col_reg[reg]] == reg)].VOC_CR.str.lower().str.count(mot.lower()) /
              data_journ[(data_journ.SIM_W2V > x) & (data_journ[col_reg[reg]] == reg)].NB_MOTS)  # fréq relative de mot de chaque article de la région "region"
        nbDocsMot = data_journ[(data_journ.SIM_W2V > x)].VOC_CR.str.lower().str.count(mot.lower())
        nbDocsMot = nbDocsMot[nbDocsMot > 0].count()  # nombre de documents qui parlent de SA, et tq mot y figure
        nbDocs = data_journ[(data_journ.SIM_W2V > x)].shape[0]  # nombre de documents qui parlent de SA
        idf = np.log((1 + nbDocs) / (1 + nbDocsMot))
        tfidf = tf * idf
        tfidf_moy = tfidf.mean()
        dict_occur[mot] = round(tfidf_moy, 6)
    top10 = dict(Counter(dict_occur).most_common(30))
    data_df = pd.DataFrame(top10.items(), columns=['MOT', 'FREQUENCE'])
    # data_df['Exps'] = data_df['MOT'] + " (" + data_df['FREQUENCE'].astype(str) + ")"
    print("Top5 voc CR\n", data_df['MOT'].head())
    data_df['MOT'].to_excel(excel_writer="result_region/top_10_cr_" + reg + ".xlsx",index=False)
    data_cloud = data_df.set_index('MOT').to_dict()['FREQUENCE']
    wordcloud = WordCloud(background_color="white", width=3000, height=2000).generate_from_frequencies(data_cloud)
    wordcloud.recolor(color_func=black_color_func)
    plt.axis("off")
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.savefig('result_region/Wordcloud_' + reg + '_CR.png')

### comparaison SA

## graphe sca
lab=["SCA"]
fig,ax=plt.subplots(figsize=(8,8))
plt.plot(data_epa.ANNEE, data_epa.sca, label=lab[0], color="black")
plt.legend(loc="best", prop={'size': 12})
plt.xlabel("Year", fontsize=15)
plt.xticks(np.arange(2009, 2019, 1))
plt.title("Evolution du FCS de 2009 à 2018")
plt.grid(True)
plt.savefig("result_annee/graphe_epa.png")

## graphe proportion articles SA
data_tmp = pd.read_excel("G:/data_hugo/doctorat/donnees_BF/vars_expl_sa/code_camille/resultats_traites/result_annee/prop_sa_annees.xlsx")
lab=["Proportion d'articles SA"]
fig,ax=plt.subplots(figsize=(8,8))
plt.plot(data_tmp.ANNEE, data_tmp.PROP_SA ,label=lab[0], color="black")
plt.legend(loc="best", prop={'size': 12})
plt.xlabel("Year", fontsize=15)
plt.xticks(np.arange(2009, 2019, 1))
plt.title("Evolution de la proportion d'articles de thème SA de 2009 à 2018")
plt.grid(True)
plt.savefig("result_annee/graphe_prop_sa.png")

## graphe proportion articles négatifs (neg>0.1)
data_tmp = pd.read_excel("G:/data_hugo/doctorat/donnees_BF/vars_expl_sa/code_camille/resultats_traites/result_annee/neg_prop_annees.xlsx")
lab=["Proportion d'articles Neg"]
fig,ax=plt.subplots(figsize=(8,8))
plt.plot(data_tmp.ANNEE, data_tmp.PROP_NEG, label=lab[0], color="black")
plt.legend(loc="best", prop={'size': 12})
plt.xlabel("Year", fontsize=15)
plt.xticks(np.arange(2009, 2019, 1))
plt.title("Evolution de la proportion d'articles négatifs de 2009 à 2018")
plt.grid(True)
plt.savefig("result_annee/graphe_prop_neg.png")

## graphe liberté de la presse
data_tmp = pd.read_excel("G:/data_hugo/doctorat/donnees_BF/vars_expl_sa/code_camille/resultats_traites/index_liberte/RSF_indicateur_liberte_press.xlsx")
lab=["Indicateur liberté presse"]
fig,ax=plt.subplots(figsize=(8,8))
plt.plot(data_tmp.ANNEE, data_tmp.SCORE_LIB, label=lab[0], color="black")
plt.legend(loc="best", prop={'size': 12})
plt.xlabel("Year", fontsize=15)
plt.xticks(np.arange(2009, 2019, 1))
plt.title("Evolution de l'indicateur de liberté de la presse (RSF) de 2009 à 2018")
plt.grid(True)
plt.savefig("result_annee/graphe_lib_presse.png")

## graphe evolution mots
data_tmp_sa = pd.read_excel("G:/data_hugo/doctorat/donnees_BF/vars_expl_sa/code_camille/resultats_traites/result_annee/top_10_sa_annees.xlsx")
data_tmp_cr = pd.read_excel("G:/data_hugo/doctorat/donnees_BF/vars_expl_sa/code_camille/resultats_traites/result_annee/top_10_cr_annees.xlsx")
rows = []
Mots_sa = ["sécurité alimentaire", "malnutrition", "sécheresse"]
Mots_cr = ["conflit", "déplacement"]
for mot in Mots_sa:
    rows.append([float(data_tmp_sa[data_tmp_sa['Exps_' + str(annee)] == mot]['Freq_' + str(annee)]) for annee in range(2009, 2019)])
for mot in Mots_cr:
    rows.append([float(data_tmp_cr[data_tmp_cr['Exps_' + str(annee)] == mot]['Freq_' + str(annee)]) for annee in range(2009, 2019)])
df_mots = pd.DataFrame(rows, columns=[annee for annee in range(2009, 2019)], index=Mots_sa+Mots_cr)

lab=["tf-idf"]
fig,ax=plt.subplots(figsize=(8,8))
for mot in Mots_sa+Mots_cr:
    plt.plot(pd.DataFrame({'col':[annee for annee in range(2009, 2019)]}), df_mots.loc[mot, :], label=mot)
plt.legend(loc="best", prop={'size': 12})
plt.xlabel("Year", fontsize=15)
plt.xticks(np.arange(2009, 2019, 1))
plt.title("Evolution du tf-idf de 5 mots de 2009 à 2018")
plt.grid(True)
plt.savefig("result_annee/graphe_evolution_mots.png")

# graphe nombre d'articles par ans
data_journ.insert(1, 'Count', 1, True)
data_journ_group = data_journ.groupby(['ANNEE']).agg({'Count':"sum"}).reset_index()

lab=["Nombre d'articles"]
fig,ax=plt.subplots(figsize=(8,8))
plt.plot(data_journ_group.ANNEE, data_journ_group.Count, label=lab[0], color="black")
plt.legend(loc="best", prop={'size': 12})
plt.xlabel("Year", fontsize=15)
plt.xticks(np.arange(2009, 2019, 1))
plt.title("Evolution du nombre d'articles de 2009 à 2018")
plt.grid(True)
plt.savefig("result_annee/graphe_nb_articles.png")

# Graphe coocurrences sécurité alimentaire

## top cooccurrence  SA
nb_mots_sa = data_journ[data_journ.SIM_W2V>x].VOC_SA.str.lower().str.count("sécurité alimentaire")
nb_mots_sa[nb_mots_sa > 1] = 1
dict_cooccur = {}
for mot in voc_det_sa:
    nb_mots = data_journ[data_journ.SIM_W2V>x].VOC_SA.str.lower().str.count(mot.lower())
    nb_mots[nb_mots > 1] = 1
    nb_sa_mots = nb_mots_sa + nb_mots

    nb_art_sa = nb_mots_sa[nb_mots_sa == 1].count()
    nb_art_mot = nb_mots[nb_mots == 1].count()
    nb_art_sa_mot = nb_sa_mots[nb_sa_mots == 2].count()
    nb_art = data_journ[data_journ.SIM_W2V>x].shape[0]

    P_sa = nb_art_sa / nb_art
    P_mot = (nb_art_mot + 1) / nb_art
    P_sa_mot = nb_art_sa_mot / nb_art

    info_mutuelle = P_sa_mot / (P_sa * P_mot)

    dict_cooccur[mot] = round(info_mutuelle, 1)
top_sa = dict(Counter(dict_cooccur).most_common(20))

## top cooccurrence  CR
nb_mots_sa = data_journ[data_journ.SIM_W2V>x].VOC_SA.str.lower().str.count("sécurité alimentaire")
nb_mots_sa[nb_mots_sa > 1] = 1
dict_cooccur = {}
for mot in voc_det_cr:
    nb_mots = data_journ[data_journ.SIM_W2V>x].VOC_CR.str.lower().str.count(mot.lower())
    nb_mots[nb_mots > 1] = 1
    nb_sa_mots = nb_mots_sa + nb_mots

    nb_art_sa = nb_mots_sa[nb_mots_sa == 1].count()
    nb_art_mot = nb_mots[nb_mots == 1].count()
    nb_art_sa_mot = nb_sa_mots[nb_sa_mots == 2].count()
    nb_art = data_journ[data_journ.SIM_W2V>x].shape[0]

    P_sa = nb_art_sa / nb_art
    P_mot = (nb_art_mot + 1) / nb_art
    P_sa_mot = nb_art_sa_mot / nb_art

    info_mutuelle = P_sa_mot / (P_sa * P_mot)

    dict_cooccur[mot] = round(info_mutuelle, 1)
top_cr = dict(Counter(dict_cooccur).most_common(20))

top_tot = {}
top_tot.update(top_sa)
top_tot.update(top_cr)
del top_tot['sécurité alimentaire']
top_tot = {k: v for k, v in sorted(top_tot.items(), key=lambda item: item[1], reverse=True)}
top_tot = dict(Counter(top_tot).most_common(10))
print(top_tot)

# Graphe coocurrences agriculture
## top cooccurrence  SA
nb_mots_sa = data_journ[data_journ.SIM_W2V>x].VOC_SA.str.lower().str.count("agriculture")
nb_mots_sa[nb_mots_sa > 1] = 1
dict_cooccur = {}
for mot in voc_det_sa:
    nb_mots = data_journ[data_journ.SIM_W2V>x].VOC_SA.str.lower().str.count(mot.lower())
    nb_mots[nb_mots > 1] = 1
    nb_sa_mots = nb_mots_sa + nb_mots

    nb_art_sa = nb_mots_sa[nb_mots_sa == 1].count()
    nb_art_mot = nb_mots[nb_mots == 1].count()
    nb_art_sa_mot = nb_sa_mots[nb_sa_mots == 2].count()
    nb_art = data_journ[data_journ.SIM_W2V>x].shape[0]

    P_sa = nb_art_sa / nb_art
    P_mot = (nb_art_mot + 1) / nb_art
    P_sa_mot = nb_art_sa_mot / nb_art

    info_mutuelle = P_sa_mot / (P_sa * P_mot)

    dict_cooccur[mot] = round(info_mutuelle, 1)
top_sa = dict(Counter(dict_cooccur).most_common(20))

## top cooccurrence  CR
nb_mots_sa = data_journ[data_journ.SIM_W2V>x].VOC_SA.str.lower().str.count("agriculture")
nb_mots_sa[nb_mots_sa > 1] = 1
dict_cooccur = {}
for mot in voc_det_cr:
    nb_mots = data_journ[data_journ.SIM_W2V>x].VOC_CR.str.lower().str.count(mot.lower())
    nb_mots[nb_mots > 1] = 1
    nb_sa_mots = nb_mots_sa + nb_mots
    nb_sa_mots = nb_mots_sa + nb_mots

    nb_art_sa = nb_mots_sa[nb_mots_sa == 1].count()
    nb_art_mot = nb_mots[nb_mots == 1].count()
    nb_art_sa_mot = nb_sa_mots[nb_sa_mots == 2].count()
    nb_art = data_journ[data_journ.SIM_W2V>x].shape[0]

    P_sa = nb_art_sa / nb_art
    P_mot = (nb_art_mot + 1) / nb_art
    P_sa_mot = nb_art_sa_mot / nb_art

    info_mutuelle = P_sa_mot / (P_sa * P_mot)

    dict_cooccur[mot] = round(info_mutuelle, 1)
top_cr = dict(Counter(dict_cooccur).most_common(20))

top_tot = {}
top_tot.update(top_sa)
top_tot.update(top_cr)
del top_tot['agriculture']
top_tot = {k: v for k, v in sorted(top_tot.items(), key=lambda item: item[1], reverse=True)}
top_tot = dict(Counter(top_tot).most_common(10))
print(top_tot)

# afficher avec networkx sur l'autre env.
