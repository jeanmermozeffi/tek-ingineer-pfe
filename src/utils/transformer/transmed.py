import re


def to_snake_case(name: str) -> str:
    # Remplacer chaque majuscule par un trait de soulignement suivi de la lettre en minuscule
    snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
    return snake_case


def to_camel_case(snake_str: str) -> str:
    # Output: decisionTreeRegressor
    components = snake_str.split('_')
    # Convertir la première composante en minuscule, le reste en majuscule
    return components[0] + ''.join(x.title() for x in components[1:])


def to_pascal_case(snake_str: str) -> str:
    # Output: DecisionTreeRegressor
    components = snake_str.split('_')
    # Convertir chaque composante en majuscule
    return ''.join(x.capitalize() for x in components)
