#include<iostream>
#include<fstream>
#include<string>

void StrFileRead(std::string s, std::string file){
	std::ofstream f(file);
	f << s;
}
