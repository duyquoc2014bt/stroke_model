import { commonOptions, selectionFields } from "@/config/Home";
import { Form, Select } from "antd";
const { Item, useForm } = Form;

export default function Home() {
  const [form] = useForm();
  return (
    <Form
      form={form}
      layout="vertical"
      onFinish={() => {}}
      onFinishFailed={() => {}}
    >
      {selectionFields.map((m) => (
        <Item {...m} key={m.name}>
          <Select options={commonOptions} />
        </Item>
      ))}
    </Form>
  );
}
