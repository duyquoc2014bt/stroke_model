"use client";
import { commonOptions, selectionFields } from "@/config/Home";
import { Form, Select, Row, Col, Button, Input, Tag } from "antd";
import { useState } from "react";
const { Item, useForm } = Form;

export default function Home() {
  const [predicted, setPredicted] = useState("");
  const [form] = useForm();
  const handleFinish = async (value: any) => {
    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: value }),
      });
      const result = await res.json();
      setPredicted(result.prediction);
      return;
    } catch (err) {
      console.log("submit failed");
    }
  };

  return (
    <Form form={form} layout="vertical" onFinish={handleFinish}>
      <Row gutter={8} wrap>
        <Col span={8}>
          <Item
            name="gender"
            label="Gender"
            rules={[{ required: true, message: "" }]}
          >
            <Select
              options={[
                { label: "Male", value: "Male" },
                { label: "Female", value: "Female" },
              ]}
            />
          </Item>
        </Col>
        <Col span={8}>
          <Item
            name="age"
            label="Age"
            rules={[{ required: true, message: "" }]}
            getValueFromEvent={(e) => Number(e.target.value) || null}
          >
            <Input type="number" />
          </Item>
        </Col>
        <Col span={8}>
          <Item
            name="bmi"
            label="BMI"
            rules={[{ required: true, message: "" }]}
            getValueFromEvent={(e) => Number(e.target.value) || null}
          >
            <Input type="number" />
          </Item>
        </Col>
        {selectionFields.map((m) => (
          <Col span={8} key={m.name}>
            <Item {...m} rules={[{ required: true, message: "" }]}>
              <Select options={commonOptions} />
            </Item>
          </Col>
        ))}
      </Row>
      <Row justify="space-between">
        <Button type="primary" htmlType="submit">
          Submit
        </Button>
        {predicted && <Tag>Kết quả dự đoán đột quỵ: {predicted}</Tag>}
      </Row>
    </Form>
  );
}
