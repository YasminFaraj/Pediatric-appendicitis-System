import os
import sys
import warnings
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

MODEL_DIR = 'Models'
COLUMN_INFO_FILE = 'app_data.xlsx'
PATIENT_FILE = 'patients.xlsx'
TARGETS = ['Diagnosis', 
           'Management', 
           'Severity']

#pré-processamento
def preprocess_input(input_series, reference_df, expected_columns, targets, full_column_list):
    df = pd.DataFrame([input_series], columns=full_column_list)

    for col in reference_df.columns:
        if col in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(reference_df[col]) and not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif pd.api.types.is_bool_dtype(reference_df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                    df[col] = df[col].astype(bool)
            except Exception:
                continue

    #preenche valores ausentes
    for col in reference_df.select_dtypes(include=np.number).columns:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(reference_df[col].median(), inplace=True)

    for col in reference_df.select_dtypes(include='bool').columns:
        if col in df.columns:
            df[col] = df[col].astype(int)

    cat_cols = [col for col in reference_df.select_dtypes(include='object').columns if col not in targets]
    for col in cat_cols:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(reference_df[col].mode()[0], inplace=True)

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    for missing in set(expected_columns) - set(df.columns):
        df[missing] = 0

    return df[expected_columns]

#previsão com modelos salvos
def predict_all_targets(patient_series, models_dict, reference_df, columns_dict, full_column_list):
    predictions = {}
    for target in TARGETS:
        x_input = preprocess_input(
            patient_series,
            reference_df,
            columns_dict[target.lower()],
            TARGETS,
            full_column_list
        )
        model = models_dict[target.lower()]['model']
        encoder = models_dict[target.lower()]['le']
        encoded = model.predict(x_input)
        predictions[f"Predicted {target}"] = encoder.inverse_transform(encoded)[0]
    return predictions

#leitura dos modelos e encoders
def load_models():
    if not os.path.exists(MODEL_DIR):
        sys.exit(f"Erro: diretório '{MODEL_DIR}' não encontrado.")

    try:
        return {
            'diagnosis': {
                'model': joblib.load(f"{MODEL_DIR}/diagnosis_model.joblib"),
                'le': joblib.load(f"{MODEL_DIR}/leDiagnosis.joblib")
            },
            'management': {
                'model': joblib.load(f"{MODEL_DIR}/management_model.joblib"),
                'le': joblib.load(f"{MODEL_DIR}/leManagement.joblib")
            },
            'severity': {
                'model': joblib.load(f"{MODEL_DIR}/severity_model.joblib"),
                'le': joblib.load(f"{MODEL_DIR}/leSeverity.joblib")
            }
        }, {
            'diagnosis': joblib.load(f"{MODEL_DIR}/trained_X_col_diagnosis.joblib"),
            'management': joblib.load(f"{MODEL_DIR}/trained_X_col_management.joblib"),
            'severity': joblib.load(f"{MODEL_DIR}/trained_X_col_severity.joblib")
        }, joblib.load(f"{MODEL_DIR}/df_columns_origin.joblib")
    except Exception as e:
        sys.exit(f"Erro ao carregar modelos: {e}")

#carrega os dados do paciente
def load_patients(columns):
    path = os.path.join(os.getcwd(), PATIENT_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo '{PATIENT_FILE}' não encontrado em {path}")

    df = pd.read_excel(path)

    #verifica colunas obrigatórias
    required = [col for col in columns if col not in TARGETS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes: {', '.join(missing)}")

    df = df[required]
    return df

#cria DataFrame temporário
def create_reference_dataframe():
    try:
        df = pd.read_excel(COLUMN_INFO_FILE)
    except FileNotFoundError:
        sys.exit(f"Erro: '{COLUMN_INFO_FILE}' não encontrado.")

    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].median(), inplace=True)

    for col in df.select_dtypes(include='bool').columns:
        df[col] = df[col].astype(int)

    for col in [c for c in df.select_dtypes(include='object') if c not in TARGETS]:
        df[col].fillna(df[col].mode()[0], inplace=True)

    for target in TARGETS:
        df[f'{target}_encoded'] = LabelEncoder().fit_transform(df[target])

    to_drop = set()
    for col in [f'{t}_encoded' for t in TARGETS]:
        vc = df[col].value_counts()
        rare = vc[vc <= 2].index
        to_drop.update(df[df[col].isin(rare)].index)

    df.drop(index=list(to_drop), inplace=True)
    return df

#main
def main():
    models, columns_dict, original_cols = load_models()
    ref_df = create_reference_dataframe()
    print("\nSistema de Suporte Clínico")

    while True:
        option = input("\nDigite '1' para acessar pacientes ou '0' para sair: ").lower()
        if option == '0':
            print("Saindo..")
            break
        elif option == '1':
            try:
                patients = load_patients(original_cols)
                for idx, row in patients.iterrows():
                    result = predict_all_targets(row, models, ref_df, columns_dict, original_cols)
                    print(f"\nPaciente {idx+1}:")
                    for key, val in result.items():
                        print(f"  {key}: {val}")
            except Exception as e:
                print(f"Erro ao acessar pacientes: {e}")
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
    