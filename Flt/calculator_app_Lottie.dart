// calculator_app.dart
// ----------------------
// Steps:
// 1. Add dependency: lottie
// 2. Run flutter pub get
// 3. Add this file in lib/
// 4. Run your app
// dependencies:
 // lottie: ^2.7.0
// ----------------------

import 'package:flutter/material.dart';
import 'package:lottie/lottie.dart';

void main() => runApp(CalculatorApp());

class CalculatorApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: CalculatorPage(),
    );
  }
}

class CalculatorPage extends StatefulWidget {
  @override
  _CalculatorPageState createState() => _CalculatorPageState();
}

class _CalculatorPageState extends State<CalculatorPage> {
  String input = "";
  String result = "";

  // Function called when any button is pressed
  void onButtonClick(String value) {
    setState(() {
      if (value == "C") {
        input = "";
        result = "";
      } else if (value == "=") {
        try {
          // Basic demo calculation: only converts string → number
          result = (double.parse(input)).toString();
        } catch (e) {
          result = "Invalid Input";
        }
      } else {
        input += value;
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Calculator")),
      
      body: Column(
        children: [
          // Lottie animation
          Lottie.network(
            "https://assets5.lottiefiles.com/packages/lf20_k86wxpgr.json",
            height: 180,
          ),

          // Display input & result
          Text("Input: $input", style: TextStyle(fontSize: 22)),
          Text("Result: $result", style: TextStyle(fontSize: 26, fontWeight: FontWeight.bold)),

          // Buttons
          Expanded(
            child: GridView.count(
              crossAxisCount: 4,
              children: ["1","2","3","+","4","5","6","-","7","8","9","*","C","0","=","/"]
                  .map((btn) =>
                ElevatedButton(
                  onPressed: () => onButtonClick(btn),
                  child: Text(btn, style: TextStyle(fontSize: 20)),
                )
              ).toList(),
            ),
          )
        ],
      ),
    );
  }
}