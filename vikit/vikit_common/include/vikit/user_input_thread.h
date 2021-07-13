#ifndef USER_INPUT_THREAD_H
#define USER_INPUT_THREAD_H

#include <termios.h>
#include <thread>

namespace vk {

/// A class that starts its own thread and listens to the console input. The
/// console input can then be inquired using the getInput() function.
class UserInputThread
{
public:
  UserInputThread();
  ~UserInputThread();

  /// Returns the latest acquired user input. Default is set to 0.
  /// Once this function is called, the input state is reset to the default.
  char getInput();

  /// Stop the thread
  void stop();

private:

  /// Main loop that waits for new user input
  void acquireUserInput();

  /// Initialize new terminal i/o settings
  void initTermios(int echo);

  /// Restore old terminal i/o settings
  void resetTermios();

  /// Read 1 character - echo defines echo mode
  int getch_(int echo);

  /// Read 1 character without echo
  int getch();

  /// Read 1 character with echo
  int getche();

  bool stop_;
  std::thread * user_input_thread_;
  char input_;

  struct termios original_terminal_settings_;
  struct termios old_terminal_settings_, new_terminal_settings_;
};

} // end namespace vk

#endif /* USER_INPUT_THREAD_H */
