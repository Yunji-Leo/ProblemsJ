package actions;

public class TopLikedTest {
    TopLiked test = new TopLiked();

    @org.junit.Test
    public void testMethod() {
        test.isMatch("aa", "a*");
    }
}
