#ifndef ASLAM_CV_COMMON_YAML_SERIALIZATION_H_
#define ASLAM_CV_COMMON_YAML_SERIALIZATION_H_

#include <fstream>  // NOLINT
#include <list>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <aslam/common/memory.h>
#include <glog/logging.h>

#include "yaml-serialization-eigen.h"

namespace YAML {

/// \brief A function to get a value from a YAML node with non-exception error handling.
/// \param[in] node The YAML node.
/// \param[in] key The key used to dereference the node (node[key]).
/// \param[out] value The return value.
/// \returns True if the value was filled in successfully. False otherwise.
template<typename ValueType>
bool safeGet(const YAML::Node& node, const std::string& key, ValueType* value) {
  CHECK_NOTNULL(value);
  bool success = false;
  if(!node.IsMap()) {
    LOG(ERROR) << "Unable to get Node[\"" << key << "\"] because the node is not a map";
  } else {
    const YAML::Node sub_node = node[key];
    if(sub_node) {
      try {
        *value = sub_node.as<ValueType>();
        success = true;
      } catch(const YAML::Exception& e) {
        LOG(ERROR) << "Error getting key \"" << key << "\" as type "
            << typeid(ValueType).name() << ": " << e.what();
      }
    } else {
      LOG(ERROR) << "Key \"" << key << "\" does not exist";
    }
  }
  return success;
}


template <class ValueType>
struct convert<std::queue<ValueType> > {
  static Node encode(const std::queue<ValueType>& queue) {
    Node node;
    typename aslam::Aligned<std::vector, ValueType>::type tmp_v;
    typename aslam::Aligned<std::queue, ValueType>::type q_cpy = queue;
    while (!q_cpy.empty()) {
      tmp_v.push_back(q_cpy.front());
      q_cpy.pop();
    }
    node = tmp_v;
    return node;
  }

  static bool decode(const Node& node, std::queue<ValueType>& queue) {
    CHECK(node.IsSequence());
    for (size_t i = 0; i < node.size(); ++i) {
      ValueType tmp = node[i].as<ValueType>();
      queue.push(tmp);
    }
    return true;
  }
};

template <class KeyType>
struct convert<std::unordered_set<KeyType> > {
  static Node encode(const std::unordered_set<KeyType>& set) {
    Node node;
    for (const KeyType& value : set) {
      node.push_back(value);
    }
    return node;
  }

  static bool decode(const Node& node, std::unordered_set<KeyType>& set) {
    set.clear();
    CHECK(node.IsSequence());
    for (YAML::const_iterator it = node.begin(); it != node.end(); ++it) {
      set.insert(it->as<KeyType>());
    }
    return true;
  }
};

template <class KeyType, class ValueType>
struct convert<std::unordered_map<KeyType, ValueType> > {
  static Node encode(const std::unordered_map<KeyType, ValueType>& map) {
    Node node;
    for (const std::pair<KeyType, ValueType>& value : map) {
      node[value.first] = value.second;
    }
    return node;
  }

  static bool decode(const Node& node,
                     std::unordered_map<KeyType, ValueType>& map) {
    map.clear();
    for (YAML::const_iterator it = node.begin(); it != node.end(); ++it) {
      map[it->first.as<KeyType>()] = it->second.as<ValueType>();
    }
    return true;
  }
};

template <typename ObjectType>
void Save(const ObjectType& object, std::ostream* ofs) {
  CHECK_NOTNULL(ofs);
  assert(ofs->good());
  YAML::Node out;
  out = object;
  *ofs << out;
}

template <typename T>
void Save(const T& object, const std::string& filename) {
  std::ofstream ofs(filename.c_str());
  Save(object, &ofs);
}

template <typename ObjectType>
bool Load(const std::string& filename, ObjectType* object) {
  CHECK_NOTNULL(object);
  std::ifstream ifs(filename.c_str());
  if (!ifs.good()) {
    return false;
  }

  try {
    YAML::Node doc = YAML::LoadFile(filename.c_str());
    (*object) = doc.as<ObjectType>();
    return true;
  }
  catch (const std::exception& e) {  // NOLINT
    LOG(ERROR) << "Encountered exception while reading yaml " << e.what();
    return false;
  }
}
}  // namespace YAML
#endif  // ASLAM_CV_COMMON_YAML_SERIALIZATION_H_
