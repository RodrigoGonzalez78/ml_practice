#Modelar un sistema de diagnóstico médico:
#¿Una persona tiene fiebre? → podría depender de si tiene gripe y si ha viajado recientemente.
#pip install pgmpy


from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Definimos la estructura de la red
modelo = BayesianNetwork([
    ('Viajo', 'Gripe'),
    ('Gripe', 'Fiebre')
])

# P(Viajo)
cpd_viajo = TabularCPD(variable='Viajo', variable_card=2, values=[[0.7], [0.3]])

# P(Gripe | Viajo)
cpd_gripe = TabularCPD(
    variable='Gripe', variable_card=2,
    values=[
        [0.9, 0.6],  # P(Gripe=No | Viajo=No, Sí)
        [0.1, 0.4]   # P(Gripe=Sí | Viajo=No, Sí)
    ],
    evidence=['Viajo'],
    evidence_card=[2]
)

# P(Fiebre | Gripe)
cpd_fiebre = TabularCPD(
    variable='Fiebre', variable_card=2,
    values=[
        [0.8, 0.2],  # P(Fiebre=No | Gripe=No, Sí)
        [0.2, 0.8]   # P(Fiebre=Sí | Gripe=No, Sí)
    ],
    evidence=['Gripe'],
    evidence_card=[2]
)

# Agregar CPDs al modelo
modelo.add_cpds(cpd_viajo, cpd_gripe, cpd_fiebre)

# Verificar que el modelo es válido
print("¿Modelo válido?", modelo.check_model())

# Inferencia usando eliminación de variables
infer = VariableElimination(modelo)

# Probabilidad de tener fiebre si viajó
consulta = infer.query(variables=['Fiebre'], evidence={'Viajo': 1})
print(consulta)


# ¿Probabilidad de gripe si NO viajó?
print(infer.query(variables=["Gripe"], evidence={"Viajo": 0}))

# ¿Probabilidad de que haya viajado si tiene fiebre?
print(infer.query(variables=["Viajo"], evidence={"Fiebre": 1}))
