
# Diagnóstico de plantas
# Dada una planta, determinar si tiene problemas de salud basándose en síntomas observados.
def diagnosticar(hojas_amarillas, tierra_seca, manchas, insectos):
    if hojas_amarillas and tierra_seca:
        return "Diagnóstico: Falta de agua"
    elif manchas and insectos:
        return "Diagnóstico: Plaga"
    elif not hojas_amarillas and not tierra_seca and not manchas and not insectos:
        return "Diagnóstico: Planta sana"
    else:
        return "Diagnóstico: No se puede determinar con certeza"


# Ejemplo 1
print(diagnosticar(
    hojas_amarillas=True,
    tierra_seca=True,
    manchas=False,
    insectos=False
))
# ➜ Diagnóstico: Falta de agua

# Ejemplo 2
print(diagnosticar(
    hojas_amarillas=False,
    tierra_seca=False,
    manchas=True,
    insectos=True
))
# ➜ Diagnóstico: Plaga

# Ejemplo 3
print(diagnosticar(
    hojas_amarillas=False,
    tierra_seca=False,
    manchas=False,
    insectos=False
))
# ➜ Diagnóstico: Planta sana
