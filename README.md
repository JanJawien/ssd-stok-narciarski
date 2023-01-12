# ssd-stok-narciarski


example.toml jest niepotrzebny i trzeba go usunąć - program sprawdza czy ten plik istnieje, trzeba to też usunąć bo inaczej się nie odpali  
default.toml też do wyrzucenia  
Doda się taki plik z powrotem kiedy ogarniemy jak działa ładowanie tych parametrów, bo na razie sa 4 miejsca w których można je podać i nie wiadomo co program rzeczywiście bierze  
Teraz ładuje parametry albo z config.py, albo z wartości podanych w kodzie (czemu to ejst tak zrobione?????????????????????????? zastapic to)  
  
Ograniczenie prędkości jest normalnie za małe bo symulacja była przemyślana dla pieszych - teraz maksymalne prędkości to 5000m/s ale tak duże wartości psują DesiredForce (opiera się ona na max_speed)  
Prędkość będzie ograniczana przez skręcanie i jazdę w poprzek stoku - wtedy narciarz zwalnia (przyspieszenie będzie mniejsze niż opory)  



