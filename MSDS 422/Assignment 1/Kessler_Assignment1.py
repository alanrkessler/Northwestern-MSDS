# MSDS 422, Section 58, Assignment 1, Alan Kessler
# Python 3.5 on Mac OS 10.13.5
# Demonstrates visualization and exploratory data analysis on survey data

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from fa_kit import FactorAnalysis
from matplotlib.backends.backend_pdf import PdfPages

# Set to get balance of size correctly
plt.rcParams['figure.dpi'] = 100

# Load data from the csv
survey = pd.read_csv('mspa-survey-data.csv')

# List of personal preference variables
vars_pref = ['Personal_JavaScalaSpark', 'Personal_JavaScriptHTMLCSS',
             'Personal_Python', 'Personal_R', 'Personal_SAS',
             'Professional_JavaScalaSpark', 'Professional_JavaScriptHTMLCSS',
             'Professional_Python', 'Professional_R', 'Professional_SAS',
             'Industry_JavaScalaSpark', 'Industry_JavaScriptHTMLCSS',
             'Industry_Python', 'Industry_R', 'Industry_SAS']

# Format data for the box plot
survey_pref = survey.copy()[vars_pref]
survey_melt = pd.melt(survey_pref)


def pref_type(row):
    """Determine type of preference"""
    if "Personal" in row['variable']:
        return "Personal"
    elif "Professional" in row['variable']:
        return "Professional"
    else:
        return "Industry"


def lang_type(row):
    """Determine type of language/software"""
    if "Scala" in row['variable']:
        return "Java/Scala/Spark"
    elif "Python" in row['variable']:
        return "Python"
    elif "CSS" in row['variable']:
        return "JavaScript/HTML/CSS"
    elif "SAS" in row['variable']:
        return "SAS"
    else:
        return "R"


# Apply the grouping functions to the "melted" data
survey_melt['Type'] = survey_melt.apply(pref_type, axis=1)
survey_melt['Lang'] = survey_melt.apply(lang_type, axis=1)

# Box plot of personal preference by language
# Appears that R & Python lead the way
plt.figure(figsize=(10, 7))
ax = sns.boxplot(x='Lang', y='value', hue='Type', data=survey_melt)
ax.set(ylabel="Software Preference")
ax.set(xlabel="Language or Software System")
plt.title("Preference Boxplots")
plt.xticks(rotation=30)
plt.subplots_adjust(bottom=0.2)
eda = PdfPages('EDA_visuals.pdf')
eda.savefig()
plt.close()


def term_count(row):
    """Calculate terms until graduation"""
    try:
        if "Fall 2016" in str(row['Graduate_Date']):
            return 0
        elif "Winter 2017" in row['Graduate_Date']:
            return 1
        elif "Spring 2017" in row['Graduate_Date']:
            return 2
        elif "Summer 2017" in row['Graduate_Date']:
            return 3
        elif "Fall 2017" in row['Graduate_Date']:
            return 4
        elif "Winter 2018" in row['Graduate_Date']:
            return 5
        elif "Spring 2018" in row['Graduate_Date']:
            return 6
        elif "Summer 2018" in row['Graduate_Date']:
            return 7
        elif "Fall 2018" in row['Graduate_Date']:
            return 8
        elif "Winter 2019" in row['Graduate_Date']:
            return 9
        elif "Spring 2019" in row['Graduate_Date']:
            return 10
        elif "Summer 2019" in row['Graduate_Date']:
            return 11
        elif "Fall 2019" in row['Graduate_Date']:
            return 12
        elif "2020 or Later" in row['Graduate_Date']:
            return 13
        else:
            return float('NaN')
    except TypeError:
        return float('NaN')


# Apply counting function to a copy of the data
survey_features = survey.copy()
survey_features['Terms_to_Grad'] = survey_features.apply(term_count, axis=1)

# Drop key-variables from the data
survey_features.drop(['Graduate_Date', 'Other', 'RespondentID'],
                     axis=1, inplace=True)

# List language specific courses for counting
courses = ['PREDICT400', 'PREDICT401', 'PREDICT410',
           'PREDICT411', 'PREDICT413', 'PREDICT420',
           'PREDICT422', 'PREDICT450', 'PREDICT451',
           'PREDICT452', 'PREDICT453', 'PREDICT454',
           'PREDICT455', 'PREDICT456', 'PREDICT457',
           'OtherPython', 'OtherR', 'OtherSAS']


def lang_count(row, lang):
    """Count the number of courses by language"""
    count = 0
    try:
        if "Other" in row['Other' + lang]:
            count += 1
    except TypeError:
        pass
    for i in courses:
        try:
            if "(" + lang + ")" in row[i]:
                count += 1
        except TypeError:
            pass
    return count


# Apply counting functions to the data
survey_features['SAS_CNT'] = survey_features.apply(lang_count,
                                                   lang='SAS', axis=1)
survey_features['R_CNT'] = survey_features.apply(lang_count, lang='R', axis=1)
survey_features['Python_CNT'] = survey_features.apply(lang_count,
                                                      lang='Python', axis=1)

# Drop course variables
survey_features.drop(courses, axis=1, inplace=True)

# Set the columns to shorter names for plotting
survey_features.columns = ['Pers_JavaScalaSpark', 'Pers_JavaScriptHTMLCSS',
                           'Pers_Python', 'Pers_R', 'Pers_SAS',
                           'Prof_JavaScalaSpark', 'Prof_JavaScriptHTMLCSS',
                           'Prof_Python', 'Prof_R', 'Prof_SAS',
                           'Ind_JavaScalaSpark', 'Ind_JavaScriptHTMLCSS',
                           'Ind_Python', 'Ind_R', 'Ind_SAS',
                           'Python_INT', 'Foundations_DE_INT',
                           'Analytics_App_INT', 'Systems_Analysis_INT',
                           'Course_CNT', 'Terms_to_Grad',
                           'SAS_CNT', 'R_CNT', 'Python_CNT']

survey_features_p = survey_features[['Pers_R', 'Prof_R', 'Ind_R',
                                     'Pers_Python',
                                     'Prof_Python', 'Ind_Python',
                                     'Pers_SAS', 'Prof_SAS', 'Ind_SAS',
                                     'Pers_JavaScalaSpark',
                                     'Prof_JavaScalaSpark',
                                     'Ind_JavaScalaSpark',
                                     'Pers_JavaScriptHTMLCSS',
                                     'Prof_JavaScriptHTMLCSS',
                                     'Ind_JavaScriptHTMLCSS']]

# Correlation heat map - modified from getting started code
# The box plots show similar results across types of preferences
# Comfirm relationships using correlations
corr = survey_features_p.corr('spearman')
top = np.zeros_like(corr, dtype=np.bool)
top[np.triu_indices_from(top)] = True
fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(corr, mask=top, cmap='coolwarm', center=0, square=True,
            linewidths=.5, cbar_kws={'shrink': 0.5},
            annot=True, annot_kws={'size': 9}, fmt='.2f')
ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right')
plt.title("Preference Rank Correlations")
plt.subplots_adjust(left=0.2)
eda.savefig()
plt.close()

# Factor analysis could represent these preferences in fewer variables
# Language preference is important to determining how class preference changes
fa_data = survey_features_p.dropna()

# Apply varimax rotated factor analysis
fa_proc = FactorAnalysis.load_data_samples(fa_data.astype(float),
                                           preproc_demean=True,
                                           preproc_scale=True)
fa_proc.extract_components()
fa_proc.find_comps_to_retain('top_n', num_keep=5)
fa_proc.reextract_using_paf()
fa_proc.rotate_components(method='varimax')

fa_fit = survey_features.copy()


def fa_scores(row, component=0):
    """Apply factors to the data"""

    # Assume there is preprocessing
    shifted = row.values - fa_proc.params_data['input_mean']
    scaled = shifted / fa_proc.params_data['input_scale']

    # Generate the factors
    factors = scaled.dot(fa_proc.comps['rot'])[0]

    return factors[component]


fa_fit['fa1'] = fa_data.apply(fa_scores, axis=1, args=(0,))
fa_fit['fa2'] = fa_data.apply(fa_scores, axis=1, args=(1,))
fa_fit['fa3'] = fa_data.apply(fa_scores, axis=1, args=(2,))
fa_fit['fa4'] = fa_data.apply(fa_scores, axis=1, args=(3,))
fa_fit['fa5'] = fa_data.apply(fa_scores, axis=1, args=(4,))

fa_fit_p1 = fa_fit[['Pers_R', 'Prof_R', 'Ind_R',
                    'Pers_Python', 'Prof_Python', 'Ind_Python',
                    'Pers_SAS', 'Prof_SAS', 'Ind_SAS',
                    'Pers_JavaScalaSpark', 'Prof_JavaScalaSpark',
                    'Ind_JavaScalaSpark', 'Pers_JavaScriptHTMLCSS',
                    'Prof_JavaScriptHTMLCSS', 'Ind_JavaScriptHTMLCSS',
                    'fa1', 'fa2', 'fa3', 'fa4', 'fa5']]

# Plot correlations to understand factor meaning
corr = fa_fit_p1.corr('spearman')
top = np.zeros_like(corr, dtype=np.bool)
top[np.triu_indices_from(top)] = True
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(corr, mask=top, cmap='coolwarm', center=0, square=True,
            linewidths=0.5, cbar_kws={'shrink': 0.5},
            annot=True, annot_kws={'size': 9}, fmt='.2f')
ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")
plt.title("Preference Factor Rank Correlation")
plt.subplots_adjust(left=0.2)
eda.savefig()
plt.close()

fa_fit_p2 = fa_fit[['fa1', 'fa2', 'fa3', 'fa4', 'fa5',
                    'Python_INT', 'Foundations_DE_INT',
                    'Analytics_App_INT', 'Systems_Analysis_INT',
                    'Course_CNT', 'Terms_to_Grad',
                    'SAS_CNT', 'R_CNT', 'Python_CNT']]
fa_fit_p2.columns = ['fa_Python', 'fa_R', 'fa_JavaScalaSpark',
                     'fa_JavaScriptHTMLCSS', 'fa_SAS',
                     'Python_INT', 'Foundations_DE_INT',
                     'Analytics_App_INT', 'Systems_Analysis_INT',
                     'Course_CNT', 'Terms_to_Grad',
                     'SAS_CNT', 'R_CNT', 'Python_CNT']

# Plot correlation to understand relationship between
# preference and course interest
corr = fa_fit_p2.corr('spearman')
top = np.zeros_like(corr, dtype=np.bool)
top[np.triu_indices_from(top)] = True
fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(corr, mask=top, cmap='coolwarm', center=0, square=True,
            linewidths=0.5, cbar_kws={'shrink': 0.5},
            annot=True, annot_kws={'size': 9}, fmt='.2f')
ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")
plt.title('Preference & Course Rank Correlation')
plt.subplots_adjust(left=0.2)
eda.savefig()
plt.close()

# Determine the top preference based on factor analysis
fa_fit['Max_Pref'] = fa_fit[['fa1', 'fa2', 'fa3', 'fa4', 'fa5']].idxmax(axis=1)


def fa_interp(row):
    """Label the factors based on correlation"""
    if row['Max_Pref'] == "fa1":
        return "Python"
    elif row['Max_Pref'] == "fa2":
        return "R"
    elif row['Max_Pref'] == "fa3":
        return "Java/Scala/Spark"
    elif row['Max_Pref'] == "fa4":
        return "JavaScript/HTML/CSS"
    else:
        return "SAS"


fa_fit['Max_Pref_Name'] = fa_fit.apply(fa_interp, axis=1)

# Plot the interest in courses by the preferred language
course_melt = pd.melt(fa_fit, id_vars=['Max_Pref_Name'],
                      value_vars=['Python_INT', 'Foundations_DE_INT',
                                  'Analytics_App_INT', 'Systems_Analysis_INT'])
fig, ax = plt.subplots(figsize=(10, 7))
ax = sns.barplot(x='variable', y='value', hue='Max_Pref_Name',
                 data=course_melt, estimator=np.mean, errwidth=0)
course_labels = ['Python', 'Data Engineering',
                 'Analytics App', 'Systems Analysis']
ax.set_xticklabels(course_labels)
ax.set(ylabel='Average Rating')
ax.set(xlabel='Course')
plt.title("Course Interest by Preferred Language")
plt.legend(bbox_to_anchor=(1, 1))
plt.subplots_adjust(right=0.75)
eda.savefig()
plt.close()

# Relationship between Python experience in program and interest
survey_scatter = fa_fit_p2.copy().dropna(subset=['Python_CNT', 'fa_Python'])

fig, ax = plt.subplots(figsize=(10, 10))
ax = sns.jointplot(x='Python_CNT', y='fa_Python', data=survey_scatter)
ax.set_axis_labels('Number of Python Courses Taken',
                   'Python Preference Factor')
eda.savefig()
plt.close()

fa_py = survey_scatter['fa_Python'].astype(float).values.reshape(-1, 1)
py_cnt = survey_scatter['Python_CNT'].astype(float).values.reshape(-1, 1)

# Standard scaler
survey_scatter['fa_Python_SS'] = StandardScaler().fit_transform(fa_py)
survey_scatter['Python_CNT_SS'] = StandardScaler().fit_transform(py_cnt)

fig, ax = plt.subplots(figsize=(10, 10))
ax = sns.jointplot(x='Python_CNT_SS', y='fa_Python_SS', data=survey_scatter)
ax.set_axis_labels("Number of Python Courses Taken (Standard Scaler)",
                   "Python Preference Factor (Standard Scaler)")
eda.savefig()
plt.close()

# MaxAbsScaler
survey_scatter['fa_Python_MAS'] = MaxAbsScaler().fit_transform(fa_py)
survey_scatter['Python_CNT_MAS'] = MaxAbsScaler().fit_transform(py_cnt)

fig, ax = plt.subplots(figsize=(10, 10))
ax = sns.jointplot(x='Python_CNT_MAS', y='fa_Python_MAS', data=survey_scatter)
ax.set_axis_labels("Number of Python Courses Taken (Max Abs Scaler)",
                   "Python Preference Factor (Max Abs Scaler)")
eda.savefig()
eda.close()
plt.close()
