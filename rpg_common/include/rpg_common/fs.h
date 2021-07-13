#pragma once

#include <string>
#include <vector>

namespace rpg_common
{
namespace fs
{

void concatenateFolderAndFileName(const std::string& folder,
                                  const std::string& file_name,
                                  std::string* path);

std::string concatenateFolderAndFileName(const std::string& folder,
                                         const std::string& file_name);

bool fileExists(const std::string& path);
bool pathExists(const std::string& path);

void splitPathAndFilename(
    const std::string& str, std::string* path, std::string* filename);

// Returns full paths.
void getFilesAndSubfolders(const std::string& path,
                           std::vector<std::string>* files,
                           std::vector<std::string>* folders);

}  // namespace fs
}  // namespace rpg_common

namespace rpg = rpg_common;
