import 'package:flutter/material.dart';

void main() => runApp(ListViewApp());

class ListViewApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(home: ListScreen(), debugShowCheckedModeBanner: false);
  }
}

class ListScreen extends StatelessWidget {
  final items = List.generate(20, (i) => "Item ${i + 1}");

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("List View")),
      body: ListView.builder(
        itemCount: items.length,
        itemBuilder: (context, index) =>
            ListTile(title: Text(items[index])),
      ),
    );
  }
}