#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<assert.h>

std::string StrFileRead(const std::string &file){
	std::ifstream f(file);
	assert(f.is_open());
	std::string str((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
	return str;
}
