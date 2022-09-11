import Message from "./Message";
import Grid from "@mui/material/Grid";

const Messages = ({ messages }) => {
  return (
    <Grid>
      {messages.map((m, i) => (
        <Message key={"ID" + i} message={m} />
      ))}
    </Grid>
  );
};

export default Messages;
