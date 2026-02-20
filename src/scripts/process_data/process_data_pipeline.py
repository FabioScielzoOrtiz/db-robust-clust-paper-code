import subprocess
import os, sys

script_path = os.path.dirname(os.path.abspath(__file__))

scripts = [
    os.path.join(script_path, "process_bikes.py"),
    os.path.join(script_path, "process_dubai_houses.py"),
    os.path.join(script_path, "process_german_credit.py"),
    os.path.join(script_path, "process_heart disease.py"),
    os.path.join(script_path, "process_kc_houses.py"),
    os.path.join(script_path, "process_students_performance.py"),

]

def run_pipeline():
    print("--- Iniciando Pipeline con Subprocess ---")

    # 1. Copiamos las variables de entorno actuales
    my_env = os.environ.copy()
    # 2. Forzamos a Python a usar UTF-8 para la entrada/salida
    my_env["PYTHONIOENCODING"] = "utf-8"

    for script in scripts:
        print(f"Ejecutando {script}...")
        
        try:
            # check=True lanza una excepción si el script falla
            result = subprocess.run(
                [sys.executable, script], 
                capture_output=True, 
                text=True, 
                check=True,
                encoding='utf-8',
                env=my_env
            )
            
            # Imprimir la salida del script (stdout)
            print(f"Salida de {script}:\n{result.stdout}")
            
        except subprocess.CalledProcessError as e:
            print(f"ERROR al ejecutar {script}.")
            print(f"Error log:\n{e.stderr}")
            break  # Detiene el pipeline si un paso falla

    print("--- Fin del proceso ---")

if __name__ == "__main__":
    run_pipeline()