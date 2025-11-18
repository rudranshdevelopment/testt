// basic_navbar.dart
// ------------------------------------------
// FEATURES:
// ✔ Bottom Navigation Bar
// ✔ Lottie Animation in each tab
// ✔ 3 Page Navigation: Home, Search, Profile
// ✔ Easy to understand UI structure

// dependencies:
//   lottie: ^2.7.0
// ------------------------------------------

import 'package:flutter/material.dart';
import 'package:lottie/lottie.dart';

void main() => runApp(NavBarApp());

class NavBarApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: BottomNavBarExample(),
    );
  }
}

class BottomNavBarExample extends StatefulWidget {
  @override
  _BottomNavBarExampleState createState() => _BottomNavBarExampleState();
}

class _BottomNavBarExampleState extends State<BottomNavBarExample> {
  int _currentIndex = 0;

  // Screens for each tab
  final List<Widget> pages = [
    HomeScreen(),
    SearchScreen(),
    ProfileScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: pages[_currentIndex],

      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        onTap: (index) {
          setState(() => _currentIndex = index);
        },
        selectedItemColor: Colors.blue,
        unselectedItemColor: Colors.grey,

        items: [
          BottomNavigationBarItem(
            icon: Lottie.network(
              "https://assets8.lottiefiles.com/packages/lf20_jk6c1n2n.json",
              height: 40,
            ),
            label: "Home",
          ),
          BottomNavigationBarItem(
            icon: Lottie.network(
              "https://assets2.lottiefiles.com/private_files/lf30_editor_tgjrra.json",
              height: 40,
            ),
            label: "Search",
          ),
          BottomNavigationBarItem(
            icon: Lottie.network(
              "https://assets1.lottiefiles.com/packages/lf20_j1adxtyb.json",
              height: 40,
            ),
            label: "Profile",
          ),
        ],
      ),
    );
  }
}

//////////////////////////////////////////////////
// HOME SCREEN
//////////////////////////////////////////////////
class HomeScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Lottie.network(
          "https://assets9.lottiefiles.com/packages/lf20_t24tpvcu.json",
          height: 250,
        ),
      ),
    );
  }
}

//////////////////////////////////////////////////
// SEARCH SCREEN
//////////////////////////////////////////////////
class SearchScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Lottie.network(
          "https://assets1.lottiefiles.com/packages/lf20_t9gkkhz4.json",
          height: 250,
        ),
      ),
    );
  }
}

//////////////////////////////////////////////////
// PROFILE SCREEN
//////////////////////////////////////////////////
class ProfileScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Lottie.network(
          "https://assets2.lottiefiles.com/packages/lf20_btu3fgtw.json",
          height: 250,
        ),
      ),
    );
  }
}