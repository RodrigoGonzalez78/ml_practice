from experta import *

# Creamos la clase para representar los hechos
class Sintomas(Fact):
    """Síntomas observados en la planta"""
    pass

# Definimos el sistema experto
class DiagnosticoPlanta(KnowledgeEngine):

    @Rule(Sintomas(hojas_amarillas=True, tierra_seca=True))
    def falta_de_agua(self):
        print("✅ Diagnóstico: Falta de agua")

    @Rule(Sintomas(manchas=True, insectos=True))
    def plaga(self):
        print("✅ Diagnóstico: Plaga")

    @Rule(Sintomas(hojas_amarillas=False, tierra_seca=False, manchas=False, insectos=False))
    def sana(self):
        print("✅ Diagnóstico: Planta sana")

    @Rule(AS.fact << Sintomas())
    def desconocido(self, fact):
        print("⚠️ Diagnóstico: No se pudo determinar con certeza.")
        print(f"  Hechos proporcionados: {fact}")


# Instanciamos el motor de inferencia
engine = DiagnosticoPlanta()

# Activamos el motor
engine.reset()

# Ejemplo: planta con hojas amarillas y tierra seca
engine.declare(Sintomas(hojas_amarillas=True, tierra_seca=True, manchas=False, insectos=False))
engine.run()

# Caso 2: Tiene manchas e insectos → plaga
engine.reset()
engine.declare(Sintomas(hojas_amarillas=False, tierra_seca=False, manchas=True, insectos=True))
engine.run()

# Caso 3: Sin síntomas → sana
engine.reset()
engine.declare(Sintomas(hojas_amarillas=False, tierra_seca=False, manchas=False, insectos=False))
engine.run()

# Caso 4: síntomas ambiguos → no determinable
engine.reset()
engine.declare(Sintomas(hojas_amarillas=True, tierra_seca=False, manchas=False, insectos=False))
engine.run()
