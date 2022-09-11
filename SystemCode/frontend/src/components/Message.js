import { BiBot } from "react-icons/bi";
import { Box, Grid } from "@mui/material";
import Paper from "@mui/material/Paper";

const Message = ({ message }) => {
  return (
    <>
      <Grid container direction="column" alignContent="flex-start">
        <Grid item alignItems="flex">
          <BiBot />
          <span>{message.file.name}</span>
        </Grid>
        <Grid item display="inline-flex">
          <Paper>
            <Box>
              <img width={400} src={message.preview}></img>
              <p>Type: {message.response.type}</p>
              <p>Probability: {message.response.probability}</p>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </>
  );
};

export default Message;
