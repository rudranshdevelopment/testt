// login_page.dart
// ---------------------
// Features included:
// - Email input
// - Password input
// - Lottie animation
// - Login button

// dependencies:
//   lottie: ^2.7.0
// ---------------------

import 'package:flutter/material.dart';
import 'package:lottie/lottie.dart';

void main() => runApp(LoginApp());

class LoginApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: LoginPage(),
    );
  }
}

class LoginPage extends StatelessWidget {
  final TextEditingController emailCtrl = TextEditingController();
  final TextEditingController passCtrl = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Padding(
        padding: EdgeInsets.all(20),

        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [

            // Lottie animation
            Lottie.network(
              "https://assets9.lottiefiles.com/packages/lf20_jcikwtux.json",
              height: 200,
            ),

            // Email input
            TextField(
              controller: emailCtrl,
              decoration: InputDecoration(labelText: "Email"),
            ),

            // Password input
            TextField(
              controller: passCtrl,
              obscureText: true,
              decoration: InputDecoration(labelText: "Password"),
            ),

            SizedBox(height: 20),

            ElevatedButton(
              onPressed: () {},
              child: Text("Login"),
            ),
          ],
        ),
      ),
    );
  }
}