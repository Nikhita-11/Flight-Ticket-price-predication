from setuptools import find_packages, setup
from typing import List 

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)-> List[str]: 
    requirements = []
    with open(file_path) as file_obj: 
        requirements = file_obj.readlines()
        requirements = [req.replace ("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements: 
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name = "Flight_Ticket_Price_Prediction", 
    version = "0.0.1", 
    author = "Sagnik Sarkar", 
    description= "An End to End ML Project to Predict The Price Of Flights",
    author_email= "sagnik.sarkar1996@gmail.com", 
    install_requires = get_requirements("requirements.txt"),
    packages= find_packages()
    )