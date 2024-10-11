from setuptools import setup, find_packages

HYPEN_E_DOT = '-e .'
def get_requirements(file_path):
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
            



setup(
    name='MedicalCostExpense',
    version='0.0.1',
    author='Latha',
    author_email='lathacharugundland@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)