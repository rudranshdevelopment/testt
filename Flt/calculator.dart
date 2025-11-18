import 'package:flutter/material.dart';

void main() => runApp(CalculatorApp());

class CalculatorApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: CalculatorScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class CalculatorScreen extends StatefulWidget {
  @override
  _CalculatorScreenState createState() => _CalculatorScreenState();
}

class _CalculatorScreenState extends State<CalculatorScreen> {
  String output = "0";
  String input = "";

  void buttonPressed(String value) {
    setState(() {
      if (value == "C") {
        input = "";
        output = "0";
      } else if (value == "=") {
        try {
          output = (double.parse(input)).toString();
        } catch (e) {
          output = "Error";
        }
      } else {
        input += value;
        output = input;
      }
    });
  }

  Widget buildButton(String label) {
    return Expanded(
      child: ElevatedButton(
        onPressed: () => buttonPressed(label),
        child: Text(label, style: TextStyle(fontSize: 25)),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Calculator")),
      body: Column(
        children: [
          Container(
            padding: EdgeInsets.all(20),
            alignment: Alignment.centerRight,
            child: Text(output, style: TextStyle(fontSize: 40)),
          ),
          Expanded(child: Divider()),
          Column(
            children: [
              Row(children: [buildButton("7"), buildButton("8"), buildButton("9")]),
              Row(children: [buildButton("4"), buildButton("5"), buildButton("6")]),
              Row(children: [buildButton("1"), buildButton("2"), buildButton("3")]),
              Row(children: [buildButton("0"), buildButton("C"), buildButton("=")]),
            ],
          )
        ],
      ),
    );
  }
}