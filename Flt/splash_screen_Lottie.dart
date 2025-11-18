// splash_screen.dart
// ------------------------
// Steps:
// 1. Splash screen shows animation
// 2. After 3 sec → navigates to HomeScreen
// dependencies:
//   lottie: ^2.7.0
// ------------------------

import 'package:flutter/material.dart';
import 'package:lottie/lottie.dart';

void main() => runApp(SplashApp());

class SplashApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: SplashScreen(),
    );
  }
}

class SplashScreen extends StatefulWidget {
  @override
  _SplashScreenState createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {

  @override
  void initState() {
    super.initState();

    // Auto navigation after delay
    Future.delayed(Duration(seconds: 3), () {
      Navigator.push(context,
        MaterialPageRoute(builder: (_) => HomeScreen()),
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Lottie.network(
          "https://assets2.lottiefiles.com/packages/lf20_kyu7xb1v.json",
          height: 250,
        ),
      ),
    );
  }
}

// After splash → Home screen UI
class HomeScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Text("Welcome!", style: TextStyle(fontSize: 28)),
      ),
    );
  }
}